import threading
import queue
from collections import deque
import traceback
import numpy as np
import sounddevice as sd
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import sys
from std_msgs.msg import String, Bool, Int32
from scipy.io import wavfile
import time
from .Conversation_state import ConversationState
from voisis_interfaces.msg import AudioRequest, TextRequest

sys.path.append(
    "/home/voice_ws/smart-turn"
)
import inference
from record_and_predict import SileroVAD, ensure_model

class smart_node(Node):

    def __init__(self):
        super().__init__("smart_vad")
        self.get_logger().info("Iniciando SmartVAD...")

        # Configuración de audio
        self.RATE = 16000
        self.CHANNELS = 6
        self.CHUNK = 512
        self.DTYPE = np.int16
        self.DEVICE = self.find_respeaker()
        self.get_logger().info(f"ReSpeaker encontrado en dispositivo {self.DEVICE}")

        # Parámetros del VAD
        self.VAD_THRESHOLD = 0.4

        # Tiempo de silencio necesario para considerar
        # que terminó una intervención.
        self.STOP_MS = 700
        self.STOP_CHUNKS = int(
            (self.STOP_MS / 1000) * self.RATE / self.CHUNK
        )

        # Estado interno
        self.speech_detected = False
        self.silence_counter = 0

        # Guarda unos 300 ms anteriores al inicio de voz
        self.pre_audio_buffer = deque(
            maxlen=int(self.RATE * 0.3 / self.CHUNK)
        )

        # Guarda toda la intervención
        self.audio_buffer = deque()

        # Queue para Endpoint Prediction
        self.audio_queue = queue.Queue(maxsize=2)

        # Cargar Silero
        self.get_logger().info("Cargando Silero VAD...")
        self.vad = SileroVAD(
            ensure_model()
        )

        # Callback Groups
        self.processing_group = MutuallyExclusiveCallbackGroup()

        # Publisher
        self.audio_publisher = self.create_publisher(
            AudioRequest,
            "hearing_smart",
            10
        )
        self.warn_publisher = self.create_publisher(
            String,
            "Stop_BP",
            10
        )

        self.prompt_publisher = self.create_publisher(
            TextRequest,
            "BERT_prompt",
            10
        )

        self.current_conversation = self.create_publisher(
            Int32,
            "CurrentConversation",
            10
        )

        self.robot_subscription = self.create_subscription(
            Bool,
            "RobotSpeaking",
            self.robot_callback,
            10
        )

        self.waiting_since = None
        self.first_interaction_timeout = 60.0
        self.idle_timer = self.create_timer(
            1.0,
            self.check_first_interaction
        )

        # Hilo para Endpoint Prediction
        self.worker = threading.Thread(
            target=self.endpoint_worker,
            daemon=True
        )
        self.worker.start()

        # Stream de audio
        self.stream = sd.InputStream(
            device=self.DEVICE,
            samplerate=self.RATE,
            channels=self.CHANNELS,
            dtype="int16",
            blocksize=self.CHUNK,   # 3 × 512
            callback=self.microfono_callback
        )
        self.stream.start()

        self.interaction_id = 0
        self.state = ConversationState.IDLE
        self.get_logger().info("SmartVAD listo.")

    def find_respeaker(self):
        devices = sd.query_devices()

        for i, device in enumerate(devices):
            name = device["name"]

            if "reSpeaker" in name or "XVF3800" in name:
                return i

        raise RuntimeError(
            "No se encontró el ReSpeaker Flex XVF3800"
        )

    def robot_callback(self, msg):
        self.robot_speaking = msg.data
        
        if msg.data:
            self.state = ConversationState.SPEAKING

        else:
            self.waiting_since = time.time()
            self.state = ConversationState.WAITING_REPLY

    def check_first_interaction(self):
        if self.state != ConversationState.WAITING_REPLY:
            return

        if self.waiting_since is None:
            return

        if time.time() - self.waiting_since < self.first_interaction_timeout:
            return

        self.state = ConversationState.CHECKING_PRESENCE
        self.waiting_since = None

        msg = TextRequest()
        msg.conversation_id = self.interaction_id
        msg.text = "[CHECK_PRESENCE]"
        self.prompt_publisher.publish(msg)
        self.get_logger().info(
            "El usuario lleva un minuto sin responder. Pregunta unicamente si sigue ahi"
        )

    # Callback del micrófono                             
    def microfono_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warning(str(status))

        waveform_int16 = indata[:,1].copy()
        # indata ya viene como float32 [-1.0, 1.0]
        waveform = waveform_int16.astype(np.float32) / 32768.0
        level = np.abs(waveform).mean()
        speech_probability = self.vad.prob(waveform)

        # Convertimos a int16 solo para guardar/transmitir WAV
        # audio_int16 = (
        #     np.clip(waveform, -1.0, 1.0) * 32767
        # ).astype(np.int16)
        audio_bytes = waveform_int16.tobytes()

        # Guardamos buffer previo
        self.pre_audio_buffer.append(audio_bytes)

        # Hay voz
        if speech_probability > self.VAD_THRESHOLD:
            self.waiting_since = None
            # Guardamos si el jaime estaba hablando
            robot_was_speaking = (
                self.state == ConversationState.SPEAKING
            )
            if self.state in (
                ConversationState.IDLE,
                ConversationState.WAITING_REPLY,
                ConversationState.CHECKING_PRESENCE,
                ConversationState.SPEAKING,
            ):
                self.state = ConversationState.LISTENING
            if not self.speech_detected:
                if robot_was_speaking:
                    self.get_logger().info(
                        "Usuario interrumpio al robot"
                    )
                    msg1 = String()
                    msg1.data = "Stop"
                    self.warn_publisher.publish(msg1)
                self.audio_buffer.clear()
                self.audio_buffer.extend(
                    self.pre_audio_buffer
                )
            self.speech_detected = True
            self.silence_counter = 0
            self.audio_buffer.append(audio_bytes)
            return

        # No hay voz
        if self.speech_detected:
            self.audio_buffer.append(audio_bytes)
            self.silence_counter += 1
            if self.silence_counter >= self.STOP_CHUNKS:
                msg2 = String()
                msg2.data = "Go"
                self.warn_publisher.publish(msg2)
                self.state = ConversationState.THINKING
                full_audio = b"".join(self.audio_buffer)
                waveform = (
                    np.frombuffer(
                        full_audio,
                        dtype=np.int16
                    )
                    .astype(np.float32)
                    / 32768.0
                )
                duration = len(waveform) / self.RATE
                if duration >= 1.0:
                    self.waiting_since = time.time()
                    try:
                        self.interaction_id += 1
                        current = self.interaction_id
                        self.audio_queue.put_nowait((full_audio, current))
                    except queue.Full:
                        self.get_logger().warning(
                            "Endpoint ocupado. Audio descartado."
                        )
                self.reset_buffers()
    
    # Hilo de Endpoint Prediction
    def endpoint_worker(self):
        while rclpy.ok():
            try:
                full_audio, current = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                waveform = (
                    np.frombuffer(
                        full_audio,
                        dtype=np.int16
                    ).astype(np.float32)
                    / 32768.0
                )
                prediction = inference.predict_endpoint(
                    waveform
                )
                if prediction["prediction"] != 1:
                    continue
                if prediction["probability"] < 0.85:
                    continue
                self.publish_audio(full_audio, current)
            except Exception:
                self.get_logger().error(
                    traceback.format_exc()
                )

    # Publicar AudioStamped
    def publish_audio(self, full_audio, current):
        filename = f"/tmp/audio_{time.time_ns()}.wav"
        samples = np.frombuffer(
            full_audio,
            dtype=np.int16
        )

        wavfile.write(
            filename,
            self.RATE,
            samples
        )

        msg = AudioRequest()
        msg.conversation_id = current
        msg.audio_path = filename
        self.audio_publisher.publish(msg)
        current_msg = Int32()
        current_msg.data = current
        self.current_conversation.publish(current_msg)

    def reset_buffers(self):
        self.audio_buffer.clear()
        self.speech_detected = False
        self.silence_counter = 0

    # Liberar recursos
    def destroy_node(self):
        self.get_logger().info(
            "Cerrando SmartVAD..."
        )
        self.stream.stop()
        self.stream.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = smart_node()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()