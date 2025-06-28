import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Any, Optional

class AkanASR:
    def __init__(self, model_path: str = "models/akan_whisper"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model = model.to(self.device)  # type: ignore
        processor_result = WhisperProcessor.from_pretrained(model_path)
        if isinstance(processor_result, tuple):
            self.processor = processor_result[0]
        else:
            self.processor = processor_result

    def transcribe(self, audio_path: str) -> str:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        # Generate transcription
        predicted_ids = self.model.generate(input_values)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Example usage
if __name__ == "__main__":
    asr = AkanASR()
    print(asr.transcribe("test_audio.wav"))  # Test with a sample file