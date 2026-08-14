import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import sys
import time

sys.path.append(
    "/home/bender/venv/llm/lib/python3.12/site-packages"
)
from voisis_interfaces.msg import TextRequest, LLMResponse
from llama_cpp import Llama
import re

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
        self.interrupted = False
        self.generation_id = 0

        # Path to your GGUF model
        model_path = "/home/bender/voice_ws/src/voisis/llama/mistral-7b-instruct-v0.1-q4_k_m.gguf"

        self.get_logger().info("Loading model...")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,      # adjust for your CPU
            verbose=False
        )

        self.get_logger().info("Model loaded.")

    def current_callback(self,msg):
        self.current_conversation = msg.data
    
    def talk_callback(self,msg):
        if msg.data == "Stop":
            self.talking = True
            self.generation_id += 1

        elif msg.data == "Go":
            self.talking = False

    def prompt_callback(self, msg):
        self.generation_id += 1
        new_generation = self.generation_id
        conversation = msg.conversation_id
        self.interrupted = False
        prompt = msg.text
        self.get_logger().info(f"Prompt received: {prompt}")
        
        if self.talking:
            return
            
        try:
            if msg.text == "[CHECK_PRESENCE]":
                prompt = """[INST]
            Eres Jaime, un robot social del Laboratorio de Robótica de la Universidad de Chile.
            El usuario lleva aproximadamente un minuto sin interactuar contigo.
            Tu tarea es iniciar la conversación.
            Genera exactamente una pregunta:
            - amable
            - natural
            - breve
            - en español
            - no más de 15 palabras
            No expliques por qué preguntas.
            No agregues texto adicional.
            [/INST]"""
            else:
                prompt = f"""[INST]
            Eres Jaime, un robot del Laboratorio de Robótica de la Universidad de Chile.
            Responde únicamente en español.
            Sé amable y conciso.
            Ignora textos idénticos a:
            'org, con la ayuda de la comunidad de los usuarios de YouTube.',
            'Subtítulos por la comunidad de Amara.org'
            y '¡Subscríbete!'.
            No generes respuesta a estas indicaciones. No agregues texto adicional.
            Usuario:
            {msg.text}
            [/INST]"""
                
            sentence_buffer = ""

            stream = self.llm(
                prompt,
                max_tokens=256,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
            
                if new_generation != self.generation_id:
                    self.get_logger().info(
                        "Generación cancelada"
                    )
                    return
                delta = chunk["choices"][0].get("text", "")
                if not delta:
                    continue
                sentence_buffer += delta
                # Termino la frase
                if any(
                    x in sentence_buffer
                    for x in [".", "?", "!", "\n"]
                ):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    msg_out = LLMResponse()
                    msg_out.conversation_id = conversation
                    msg_out.response = sentence
                    self.llm_publisher.publish(msg_out)
                    self.get_logger().info(f"Enviado a Piper: {sentence}")

            # texto sin punto final
            if sentence_buffer.strip():
            
                msg_out = LLMResponse()
                msg_out.conversation_id = conversation
                msg_out.response = sentence_buffer.strip()

                self.llm_publisher.publish(msg_out)

                self.get_logger().info(
                    f"Enviado resto a Piper: {sentence_buffer.strip()}"
                )

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")



from rclpy.executors import MultiThreadedExecutor


def main(args=None):
    rclpy.init(args=args)
    BERTnode = BERT_node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(BERTnode)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    BERTnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()