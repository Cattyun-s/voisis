import os
import wave
from piper import PiperVoice, SynthesisConfig
    

def piperfunc():
    voice_path = "/home/bender/voice_ws/src/voisis/voices/es_MX-claude-high.onnx" #check once everything is passed to ros2
    
    voice = PiperVoice.load(voice_path) #, use_cuda=True) hay que cambiar a cuda 12. cualquiera
    
    
    with open("/home/bender/voice_ws/src/voisis/llama/BERTlog.txt") as f:
        ans = f.read()
    
    with wave.open("test.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
    
        syn_config = SynthesisConfig(
            volume=0.5,  # half as loud
            length_scale=1.0,  # twice as slow
            noise_scale=1.0,  # more audio variation
            noise_w_scale=0.8,  # more speaking variation
            normalize_audio=False, # use raw audio from voice
        )
        
        voice.synthesize_wav(ans, wav_file, syn_config=syn_config)
    
    os.system("aplay test.wav")
