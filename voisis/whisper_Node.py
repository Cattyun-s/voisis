# Recibe audio de smartVAD_Node.py y lo transcribe
# para luego publicarlo en BERT_prompt como mensaje inicial

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

        self.publisher_prompt = self.create_publisher(
            TextRequest,
            "BERT_prompt",
            10
        )

        self.SAMPLE_RATE = 16000
        self.LANGUAGE = "es"
        self.MODEL_SIZE = "small"
        self.q = queue.Queue(maxsize=1)
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
        self.get_logger().info("Whisper listo")

    def has_audio(self, audio_path):
        try:
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
        try:
            if self.q.full():
                self.q.get_nowait()

            self.q.put_nowait((conversation, audio))
        except queue.Empty:
            pass

    def send_message(self):
        while rclpy.ok():
            try:
                conversation, audio = self.q.get(timeout=1)
                segments, info = self.whisper_model.transcribe(
                    audio,
                    beam_size=5,
                    language=self.LANGUAGE,
                    temperature=0.0,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.0,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    vad_parameters= {
                        "min_silence_duration_ms":500
                    }
                )

                texts = []

                for segment in segments:
                    self.get_logger().info(
                        f"segment='{segment.text}'"
                        f"no_speech={segment.no_speech_prob:.3f}"
                        f"logprob={segment.avg_logprob:.3f}"
                    )
                    if segment.no_speech_prob >= 0.6:
                        continue
                    
                    if segment.avg_logprob < -1.0:
                        continue
                    
                    text = segment.text.strip()

                    if not text:
                        continue
                    
                    texts.append(text)

                text_result = " ".join(texts).strip()

                if not text_result:
                    continue

                self.get_logger().info(
                    f"Whisper: {text_result}"
                )

                msg_out = TextRequest()
                msg_out.conversation_id = conversation
                msg_out.text = text_result

                self.publisher_prompt.publish(
                    msg_out
                )

            except queue.Empty:
                continue

            except Exception as e:
                self.get_logger().error(
                    repr(e)
                )

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