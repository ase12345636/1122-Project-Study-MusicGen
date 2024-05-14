import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor


class Compressor():
    def __init__(self, src_sample_rate: int = 48000, tgt_sample_rate: int = 32000):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_32khz")
        self.model = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.resampler = torchaudio.transforms.Resample(
            src_sample_rate, tgt_sample_rate)

    def compress(self, file_path: str):
        audio, _ = torchaudio.load(file_path)
        audio = torch.mean(audio, dim=0)

        input = self.processor(
            audio, sampling_rate=32000, return_tensors="pt")

        output = self.model.encode(
            input["input_values"], input["padding_mask"]).audio_codes

        return output
