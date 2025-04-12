"""
Real-time Akan speech-to-text using trained model
Integrates with FastAPI backend
"""

import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class AkanASR:
    def __init__(self, model_path: str = "models/akan_whisper"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model.to(self.device)

    def transcribe(self, audio_path: str) -> str:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_values.to(self.device)
        
        # Generate transcription
        predicted_ids = self.model.generate(inputs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Example usage
if __name__ == "__main__":
    asr = AkanASR()
    print(asr.transcribe("test_audio.wav"))  # Test with a sample file