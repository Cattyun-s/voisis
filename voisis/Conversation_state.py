from enum import Enum, auto

class ConversationState(Enum):
    IDLE = auto()                   # Esperando por usuario
    LISTENING = auto()              # Escuchando
    THINKING = auto()               # Generando respuesta (whisper/Bert)
    SPEAKING = auto()               # Leyendo la respuesta generada (Piper)
    WAITING_REPLY = auto()          # Esperando respuesta del usuario
    CHECKING_PRESENCE = auto()     # Preguntar "¿Sigues ahi?"