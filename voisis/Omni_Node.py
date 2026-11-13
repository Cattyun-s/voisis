import rclpy
from rclpy.node import Node
import os
import wave
import sys
# sys.path.append(
#     "/home/bender/venv/llm/lib/python3.12/site-packages"
# )
import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd
import subprocess
from std_msgs.msg import String, Bool, Int32, UInt8MultiArray
from voisis_interfaces.msg import LLMResponse
import torch
import soundfile as sf
from omnivoice import OmniVoice


class OmniVoice_Node(Node):

    def __init__(self):
        super().__init__('omni_tts_node')
        self.model_name = "k2-fsa/OmniVoice"
        self.get_logger().info(f"Cargando OmniVoice: {self.model_name}")

        self.voice =OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda",
            dtype=torch.float16
        )

        self.get_logger().info("OmniVoice cargado correctamente")

        self.sample_rate = 24000 #estaba cmo 24000 #cambio a 16000 no hizo nada 
        self.voice_instruct = "male, very low pitch"

        # Archivo temporal de audio
        # self.audio_path = "/tmp/voisis_omni.wav"
        
        try:
            self.device_id = self.find_respeaker_output()
            info = sd.query_devices(self.device_id)
            self.out_channels = max(1, min(info["max_output_channels"], 2))
            self.get_logger().info(f"Salida ReSpeaker encontrada en indice: {self.device_id}, canales={self.out_channels}")
        except Exception as e:
            self.get_logger().warning(f"ReSpeaker no encontrado: {e}")
            self.device_id = None
            self.out_channels = 1
            
        self.ignore_queue = False
        self.stop_requested = False
        # self.player = None

        self.current_conversation = 0

        self.BERT_subscription = self.create_subscription(
            LLMResponse,
            'BERT_response',
            self.listener_callback,
            10
        )

        self.RobotSpeaking = self.create_publisher(
            Bool,
            'RobotSpeaking',
            10
        )
        
        # self.BERT_subscription --> prevent unused variable warning
        self.stop_subscription = self.create_subscription(
            String,
            "Stop_BP",
            self.talk_callback,
            10
        )

        self.current_subscription = self.create_subscription(
            Int32,
            "CurrentConversation",
            self.current_callback,
            10
        )
        
    def find_respeaker_output(self):
        devices = sd.query_devices()
        keywords = ("xvf3800", "c16k6ch", "respeaker", "respeaker flex xvf3800 c16k6ch")

        # Prioridad 1: Buscar por nombre del dispositivo
        for i, dev in enumerate(devices):
            name = dev["name"].lower()
            if any(k in name for k in keywords) and dev["max_output_channels"] > 0:
                self.get_logger().info(f"Dispositivo encontrado por nombre: {dev['name']} (indice {i})")
                return i

        # Prioridad 2: Buscar el enlace directo de hardware ALSA (hw:2,0)
        for i, dev in enumerate(devices):
            # if "hw:2,0" in dev["name"] and dev["max_output_channels"] > 0:
            #     self.get_logger().info(
            #         f"Dispositivo de salida fijado en hw:2,0 (Índice {i})"
            #     )
            name = dev["name"].lower()
            if any(k in name for k in ["pulse", "pipewire", "default"]) and dev["max_output_channels"] > 0:
                self.get_logger().info(f"Usando servidor de sonido del sistema")
                return i
            
        #Diag: lista lo que ve este proceso
        self.get_logger().error(
            "Dispositivos de salida visibles: "
            + ", ".join(f"[{i}] {d['name']}" for i, d in enumerate(devices)
                        if d["max_output_channels"] > 0)
        )

        raise RuntimeError("No se encontró la salida del ReSpeaker XVF3800.")

    def current_callback(self, msg):
        self.current_conversation = msg.data

    def talk_callback(self,msg):
        if msg.data=="Stop":
            self.get_logger().info("Stop recibido.")
            self.stop_requested=True

        elif msg.data=="Go":
            self.get_logger().info("Go recibido")
            self.stop_requested=False

    def listener_callback(self, msg):
        self.get_logger().info(f"RECIBIDO BERT_: conversation_id={msg.conversation_id}")

        if msg.conversation_id != self.current_conversation:
            self.get_logger().info(f"Descartando conversación {msg.conversation_id}; " f"actual={self.current_conversation}")
            return
        # self.get_logger().info(f"Reproduciendo conversación {msg.conversation_id}")
        self.get_logger().info(f"Texto recibido para omni: '{msg.response}'")
        self.omni_tts(msg.response)
        
    def ply_tts(self, audio):
        audio = np.squeeze(np.asarray(audio, dtype=np.float32))
        audio_16k = resample_poly(audio,2,3).astype(np.float32)
        audio_16k = np.clip(audio_16k, -1.0, 1.0)
        
        blocksize = 512
        
        try:
            with sd.OutputStream(
                device=self.device_id,
                samplerate=16000,
                channels=self.out_channels,
                dtype="float32",
                blocksize=blocksize,
            ) as stream:
                for start in range(0, len(audio_16k), blocksize):
                    if self.stop_requested:
                        self.get_logger().info("Playback interrumpido.")
                        stream.abort()
                        break
                    chunk = audio_16k[start:start + blocksize].reshape(-1,1)
                    stream.write(np.tile(chunk,(1, self.out_channels)))
        
        except Exception as e:
            self.get_logger().error(f"Error reproduciendo TTS: {e}")

    def omni_tts(self, text):
        self.get_logger().info("Entrando a omni_tts()")

        if self.stop_requested:
            self.get_logger().info("Omni detenido por Stop")
            return

        speaking_msg = Bool()
        speaking_msg.data = True
        self.RobotSpeaking.publish(speaking_msg)

        try:
            self.get_logger().info("Generando audio...")
            audio = self.voice.generate(
                text=text,
                instruct=self.voice_instruct
            )

            if self.stop_requested:
                self.get_logger().info("Stop recibido después de generar audio")
                return

            self.get_logger().info("Reproduciendo TTS...")
            self.ply_tts(audio)

        except Exception as e:
            self.get_logger().error(f"Error en Omni: {e}")

        finally:
            speaking_msg = Bool()
            speaking_msg.data = False
            self.RobotSpeaking.publish(speaking_msg)

            self.get_logger().info("OmniVoice_tts() terminado.")


def main(args=None):
    rclpy.init(args=args)

    omni_node = OmniVoice_Node()

    rclpy.spin(omni_node)

    omni_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()