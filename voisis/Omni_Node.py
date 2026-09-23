import rclpy
from rclpy.node import Node
import os
import wave
import sys
sys.path.append(
    "/home/bender/venv/llm/lib/python3.12/site-packages"
)

import subprocess
from std_msgs.msg import String, Bool, Int32
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

        self.sample_rate = 24000
        self.voice_instruct = "male, very high pitch, native Spanish speaker, Latin American Spanish accent"

        # Archivo temporal de audio
        self.audio_path = "/tmp/voisis_omni.wav"

        self.ignore_queue = False
        self.stop_requested = False
        self.player = None

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

    def current_callback(self, msg):
        self.current_conversation = msg.data

    def talk_callback(self,msg):
        if msg.data=="Stop":
            self.get_logger().info("Stop recibido.")
            self.stop_requested=True

            if self.player is not None:
                self.get_logger().info("Deteniendo reproduccion...")
                self.player.kill()
                self.player = None

        elif msg.data=="Go":
            self.get_logger().info("Go recibido")
            self.stop_requested=False

    def listener_callback(self, msg):
        self.get_logger().info(f"RECIBIDO BERT_: conversation_id={msg.conversation_id}")

        if msg.conversation_id != self.current_conversation:
            self.get_logger().info(f"Descartando conversación {msg.conversation_id}", f"actual={self.current_conversation}")
            return
        # self.get_logger().info(f"Reproduciendo conversación {msg.conversation_id}")
        self.get_logger().info(f"Texto recibido para piper: '{msg.response}'")
        self.omni_tts(msg.response)

    def omni_tts(self, text):
        self.get_logger().info("Entrando a omni_tts()")
        if self.stop_requested:
            self.get_logger().info("Omni detenido por Stop")
            return
        speaking_msg = Bool()
        speaking_msg.data =True
        self.RobotSpeaking.publish(speaking_msg)

        try:
            self.get_logger().info("Generando audio...")
            audio = self.voice.generate(
                text=text,
                language_id="spa",
                instruct=self.voice_instruct
            )

            if self.stop_requested:
                self.get_logger().info("Stop recibido despues de generar audio")
                return
            
            sf.write(
                self.audio_path,
                audio[0],
                self.sample_rate
            )

            self.get_logger().info(f"Audio generado: {self.audio_path}")

            self.player = subprocess.Popen(["aplay", self.audio_path])
            self.get_logger().info("Reproduciendo")

            self.player.wait()
            self.player = None

        except Exception as e:
            self.get_logger().error(f"Error en Omni: {e}")

        finally:
            speaking_msg = Bool()
            speaking_msg.data = False

            self.RobotSpeaking.publish(speaking_msg)
            self.get_logger().info("OmniVoice_tt() terminado.")


def main(args=None):
    rclpy.init(args=args)

    omni_node = OmniVoice_Node()

    rclpy.spin(omni_node)

    omni_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()