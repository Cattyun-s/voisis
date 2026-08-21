# https://huggingface.co/bartowski/Qwen2.5-3B-GGUF

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from rclpy.executors import MultiThreadedExecutor

import sys
import threading
import re

sys.path.append(
    "/home/voice_ws/venv/python3.12/site-packages"
)
from voisis_interfaces.msg import TextRequest, LLMResponse
from llama_cpp import Llama

class BERT_node(Node):

    def __init__(self):
        super().__init__('BERT_llm_node')

        # Publisher for responses
        self.llm_publisher = self.create_publisher(
            LLMResponse,
            'BERT_response',
            10
        )

        # Subscriber for prompts
        self.prompt_subscription = self.create_subscription(
            TextRequest,
            'BERT_prompt',
            self.prompt_callback,
            10
        )

        self.VAD_sub = self.create_subscription(
            String,
            'Stop_BP',
            self.talk_callback,
            10
        )

        self.current_subscription = self.create_subscription(
            Int32,
            "CurrentConversation",
            self.current_callback,
            10
        )

        self.current_conversation = 0
        self.talking = False
        # self.interrupted = False
        self.generation_id = 0
        self.generation_lock = threading.Lock()

        # Path to your GGUF model
        model_path = "/home/voice_ws/src/voisis/llama/Qwen2.5-3B-Q4_K_M.gguf"

        self.get_logger().info("Loading model...")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,      # adjust for your CPU
            verbose=False
        )
        self.get_logger().info(
            f"Chat format detectado: {self.llm.chat_format}"
        )
        self.get_logger().info(
            f"Chat template: {self.llm.metadata.get('tokenizer.chat_template', 'NO ENCONTRADO')}"
        )

        self.get_logger().info("Model loaded.")

    def current_callback(self,msg):
        self.current_conversation = msg.data
        self.get_logger().info(f"Current conversation: {self.current_conversation}")
    
    def talk_callback(self,msg):
        if msg.data == "Stop":
            self.get_logger().info("Stop recibido. Cancelando generacion actual.")
            self.talking = True

            with self.generation_lock:
                self.generation_id += 1

        elif msg.data == "Go":
            self.get_logger().info("Go recibido. LLM habilitado.")
            self.talking = False

    def check_generation(self, generation_id):
        with self.generation_lock:
            return generation_id == self.generation_id

    def prompt_callback(self, msg):
        if self.talking:
            self.get_logger().info("prompt ignorado porque el robot esta detenido.")
            return

        with self.generation_lock:
            self.generation_id += 1
            generation_id = self.generation_id

        conversation_id = msg.conversation_id

        self.get_logger().info(f"Prompt recibido (conversation={conversation_id}): {msg.text}")
            
        if msg.text == "[CHECK_PRESENCE]":
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Eres Jaime, un robot social del "
                        "Laboratorio de Robótica de la Universidad de Chile. "
                        "Responde únicamente en español."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "El usuario lleva aproximadamente un minuto "
                        "sin interactuar contigo. "
                        "Inicia la conversación con exactamente una pregunta "
                        "breve, amable y natural. "
                        "No uses más de 15 palabras."
                    )
                }
            ]

        else:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Eres Jaime, un robot del Laboratorio de Robótica\n"
                        "de la Universidad de Chile.\n"
                        "Tu trabajo es ser mayordomo del Laboratorio"
                        "Responde únicamente en español.\n"
                        "Sé amable, natural y conciso.\n"
                        "Responde solamente a la última pregunta del usuario.\n"
                        "No inventes conversaciones anteriores.\n"
                        "No escribas etiquetas como [INST], [/INST], "
                        "### Response, ### User o ### Assistant.\n"
                        "No hagas preguntas ficticias del usuario.\n"
                        "No continúes conversaciones imaginarias.\n"
                        "No uses emojis, emoticonos, stickers ni símbolos decorativos."
                    )
                },
                {
                    "role": "user",
                    "content": msg.text
                }
            ]

        sentence_buffer = ""

        try:

            self.get_logger().info(f"Iniciando generacion {generation_id}")

            stream = self.llm.create_chat_completion(
                messages=prompt,
                max_tokens=80,
                temperature=0.2,
                stop=[
                    "<|im_start|>",
                    "<|im_end|>"
                ],
                stream=True
            )

            for chunk in stream:
            
                if not self.check_generation(generation_id):
                    self.get_logger().info(
                        f"Generación {generation_id} cancelada"
                    )
                    return
                delta = chunk["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                sentence_buffer += delta
                # Termino la frase
                while True:
                    match = re.search(
                        r'(.+?[.!?\n])',
                        sentence_buffer,
                        re.DOTALL
                    )

                    if not match:
                        break

                    sentence = match.group(1).strip()
                    sentence_buffer = sentence_buffer[match.end(): ]

                    if not sentence:
                        continue

                    self.publish(
                        sentence,
                        conversation_id
                    )

            if (sentence_buffer.strip() and self.check_generation(generation_id)):
                self.publish(
                    sentence_buffer.strip(),
                    conversation_id
                    )

            self.get_logger().info(f"Generation {generation_id} terminada")

        except Exception as e:
            self.get_logger().error(f"Inference failed: {type(e).__name__}: {e}")

    def clean_response(self, text):

        # Eliminar emojis y pictogramas
        text = re.sub(
            r'[\U0001F300-\U0001FAFF'
            r'\U00002700-\U000027BF'
            r'\U0001F1E6-\U0001F1FF]+',
            '',
            text
        )

        # Eliminar etiquetas de chat
        text = re.sub(
            r'<\|im_start\|>|<\|im_end\|>',
            '',
            text
        )

        # Eliminar etiquetas que el modelo pueda generar
        text = re.sub(
            r'\[/?INST\]',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'###\s*(Response|User|Assistant|System)',
            '',
            text,
            flags=re.IGNORECASE
        )

        # Eliminar "user", "assistant", "system"
        text = re.sub(
            r'(?i)\b(user|assistant|system)\b\s*:?',
            '',
            text
        )

        # Limpiar espacios
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
    def publish(self, text, conversation_id):
    
        text = self.clean_response(text)
    
        if not text:
            return
    
        msg_out = LLMResponse()
        msg_out.conversation_id = conversation_id
        msg_out.response = text
    
        self.llm_publisher.publish(msg_out)
    
        self.get_logger().info(
            f"Enviado a Piper "
            f"(Conversation={conversation_id}): {text}"
        )
    
def main(args=None):
    rclpy.init(args=args)
    BERTnode = BERT_node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(BERTnode)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        BERTnode.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()