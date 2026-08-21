#python script que corre el llm
#la idea es que inicie el llm y luego este recibiendo input (texto) hasta que se termine el proceso (para terminar hay palabras clave o cn ctrl+c)
#cuando parta el llm le da inmediatamente la indicacion de que no conteste con mas de 2 parrafos a menos que se solicite especificar/profundizar
#la idea es que desde aca se reciba un input y la respuesta quede guardada en una variable para que se le pueda entregar a piper

from llama_cpp import Llama
import re

# Initialize the model
llm = Llama(
    model_path="/home/voice_ws/src/voisis/llama/mistral-7b-instruct-v0.1-q4_k_m.gguf",
    n_ctx=1024,       # Context window size
    n_threads=8,      # Number of CPU threads to use
    n_gpu_layers=-1,   # Number of layers to offload to GPU (set to 0 for CPU only). og is 35, changed to 0
    n_batch=512,
    use_mmap=True,
    use_mlock=True,
    verbose=False
)

messages = [
    {"role": "system", "content": "Responde en dos o menos párrafos. Siempre responde en español."}
]

BERTans = []

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]: #palabras para que se apague/termine
        print("Exiting...")
        break

    messages.append({"role": "user", "content": user_input})

    assistant_message = ""
    sentence_buffer = ""
    with open("/home/voice_ws/voisis/llama/BERTlog.txt", "w") as f:
        f.write('')
    
    print("\nAssistant: ", end="", flush=True)

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=256,
        temperature=0.7,
    )

    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")

        if delta:
            assistant_message += delta
            sentence_buffer += delta

            # Split on sentence endings
            parts = re.split(r'([.!?\n])', sentence_buffer)

            # Print complete sentences
            completed = ""
            
            for i in range(0, len(parts) - 1, 2):
                completed += parts[i] + parts[i + 1]
            
            if completed:
                print(completed, end="", flush=True)
               
                with open("/home/voice_ws/voisis/llama/BERTlog.txt", "w") as f:
                    f.write(completed)
                # Keep unfinished remainder
                sentence_buffer = "".join(parts[len(parts) - (len(parts) % 2):])

    # Print any remaining partial text
    if sentence_buffer.strip():
        print(sentence_buffer, end="", flush=True)

        with open("/home/voice_ws/voisis/llama/BERTlog.txt", "w") as f:
            f.write(sentence_buffer)


    print()



    messages.append({"role": "assistant", "content": assistant_message})

