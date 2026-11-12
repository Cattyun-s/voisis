# Recibe audio de smartVAD_Node.py y lo transcribe
# para luego publicarlo en BERT_prompt como mensaje inicial
import os
import re
import rclpy
from rclpy.node import Node
from faster_whisper import WhisperModel
import threading
import queue
from std_msgs.msg import Int32
from voisis_interfaces.msg import AudioRequest, TextRequest
import numpy as np
from scipy.io import wavfile

class whisper_node(Node):
    def __init__(self):
        super().__init__("whisper_write_node")

        self.subscription = self.create_subscription(
            AudioRequest,
            "hearing_smart",
            self.transcription,
            10
        )
        
        self.current_sub = self.create_subscription(
            Int32,
            "CurrentConversation",
            self.current_callback,
            10
        )

        self.publisher_prompt = self.create_publisher(
            TextRequest,
            "BERT_prompt",
            10
        )

        self.SAMPLE_RATE = 16000
        self.LANGUAGE = "es"
        self.MODEL_SIZE = "small"
        self.q = queue.Queue(maxsize=5)
        
        self.current_conversation = 0
        self.conv_lock = threading.Lock()
        
        self.HALLUCINATION_PATTERNS = [
            r"subt[ií]tulos",
            r"comunidad de amara",
            r"gracias por ver",
            r"suscr[ií]bete",
            r"¡Suscr[ií]bete!",
            r"nos vemos en el pr[oó]ximo",
            r"transcripci[oó]n realizada por",
            r"continuar[aá]",
            r"[\.]{3,}",
        ]
        
        self.get_logger().info("Cargando Whisper...")

        self.whisper_model = WhisperModel(
            self.MODEL_SIZE,
            device="cuda",
            compute_type="float16"
        )

        self.worker = threading.Thread(
            target=self.send_message,
            daemon=True
        )

        self.worker.start()
        self.get_logger().info("Whisper listo con el filtro de aluciinaciones")

    def current_callback(self, msg):
        with self.conv_lock:
            self.current_conversation = msg.data
            
    def hallucination(self,text):
        # Detecta alucinaciones
        clean_text = text.lower().strip()
        
        if len(clean_text) <= 1:
            return True
        
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, clean_text):
                return True
            
        words = clean_text.split()
        if len(words) >= 3 and len(set(words)) == 1:
            return True
        
        return False

    def has_audio(self, audio_path):
        # Valida si el archivo .wav contiene señal util (RMS suficiente)
        try:
            if not os.path.exists(audio_path):
                return False
            rate, audio = wavfile.read(audio_path)
            if len(audio) == 0:
                return False
            audio_float = audio.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            self.get_logger().info(
                f"Audio RMS: {rms:.6f}"
            )
            return rms > 0.005

        except Exception as e:
            self.get_logger().error(
                f"Error analizando audio: {e}"
            )
            return False

    def transcription(self, msg):
        conversation = msg.conversation_id
        audio = msg.audio_path
        
        with self.conv_lock:
            if conversation < self.current_conversation:
                self.get_logger().warning(f"Descartando audio obsoleto ID={conversation} (Actual={self.current_conversation})")
                self._cleanup_file(audio)
                return
        try:
            if self.q.full():
                old_conv, old_audio = self.q.get_nowait()
                self._cleanup_file(old_audio)

            self.q.put_nowait((conversation, audio))
        except queue.Empty:
            pass

    def send_message(self):
        while rclpy.ok():
            try:
                conversation, audio = self.q.get(timeout=1)
                
                with self.conv_lock:
                    if conversation < self.current_conversation:
                        self.get_logger().info(f"Cancelando transcripcion ID={conversation} por nueva interaccion ID={self.current_conversation}")
                        self._cleanup_file(audio)
                        continue
                
                if not self.has_audio(audio):
                    self.get_logger().warning(f"Audio omitido : {audio}")
                    self._cleanup_file(audio)
                    continue
                
                segments, info = self.whisper_model.transcribe(
                    audio,
                    beam_size=5,
                    language=self.LANGUAGE,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False
                    # no_speech_threshold=0.0,
                    # log_prob_threshold=-1.0,
                    # compression_ratio_threshold=2.4,
                    # vad_parameters= {
                        # "min_silence_duration_ms":500
                    # }
                )

                texts = []

                for segment in segments:
                    self.get_logger().info(
                        f"segment='{segment.text}'"
                        f"no_speech={segment.no_speech_prob:.3f}"
                        f"logprob={segment.avg_logprob:.3f}"
                    )
                    if segment.no_speech_prob >= 0.8:
                        continue
                    
                    if segment.avg_logprob < -2.0:
                        continue
                    
                    text = segment.text.strip()

                    if text:
                        texts.append(text)

                text_result = " ".join(texts).strip()
                
                with self.conv_lock:
                    current_activate = self.current_conversation

                if text_result and conversation >= current_activate:
                    self.get_logger().info(f"Whisper [{conversation}]: {text_result}")

                    msg_out = TextRequest()
                    msg_out.conversation_id = conversation
                    msg_out.text = text_result

                    self.publisher_prompt.publish(msg_out)
                
                else:
                    self.get_logger().warning(f"Whisper no produjo texto paraa la conversacion {conversation}")

                self._cleanup_file(audio)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error en worker Whisper: {repr(e)}")

    def _cleanup_file(self,path):
        # Elimina el archivo temp .wav para no saturar el disco
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            self.get_logger().warning(f"No se pudo eliminar {path}: {e}")

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = whisper_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()