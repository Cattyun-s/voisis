import rclpy
from rclpy.node import Node
import os
import wave
import sys
sys.path.append(
    "/home/voice_ws/venv/lib/python3.12/site-packages"
)

from piper import PiperVoice, SynthesisConfig
import subprocess
from std_msgs.msg import String, Bool, Int32
from voisis_interfaces.msg import AudioRequest, LLMResponse


class Piper_Node(Node):

    def __init__(self):
        super().__init__('Piper_tts_node')

        self.voice_path = "/home/voice_ws/src/voisis/voices/es_MX-claude-high.onnx" #check once everything is passed to ros2
    
        self.voice = PiperVoice.load(self.voice_path) #, use_cuda=True) hay que cambiar a cuda 12. cualquiera

        self.ignore_queue = False
        self.stop_requested = False
        self.player = None

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
        
        self.BERT_subscription # prevent unused variable warning
        self.stop_subscription = self.create_subscription(
            String,
            "Stop_BP",
            self.talk_callback,
            10
        )

        self.current_conversation = 0

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
            self.stop_requested=True
            if self.player is not None:
                self.player.kill()
                self.player = None
        elif msg.data=="Go":
            self.stop_requested=False

    def listener_callback(self, msg):
        if msg.conversation_id != self.current_conversation:
            self.get_logger().info(f"Descartando conversación {msg.conversation_id}")
            return
        self.get_logger().info(f"Reproduciendo conversación {msg.conversation_id}")
        self.Piper_tts(msg.response)

    def Piper_tts(self, ans):
        msg1 = Bool()
        msg1.data = True
        self.RobotSpeaking.publish(msg1)

        with wave.open("test.wav", "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.voice.config.sample_rate)

            syn_config = SynthesisConfig(
                volume=0.5,  # half as loud
                length_scale=1.0,  # twice as slow
                noise_scale=1.0,  # more audio variation
                noise_w_scale=0.8,  # more speaking variation
                normalize_audio=False, # use raw audio from voice
            )

            if self.stop_requested:
                return

            self.voice.synthesize_wav(ans, wav_file, syn_config=syn_config)

        self.player = subprocess.Popen(
            ["aplay", "test.wav"]
        )
        self.player.wait()

        msg = Bool()
        msg.data = False
        self.RobotSpeaking.publish(msg)

        


def main(args=None):
    rclpy.init(args=args)

    piper_node = Piper_Node()

    rclpy.spin(piper_node)

    piper_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()