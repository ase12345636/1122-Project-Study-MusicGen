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
        audio = self.resampler(audio)
        input = self.processor(audio, return_tensors="pt")

        output_codebook_index = self.model.encode(
            input["input_values"], input["padding_mask"]).audio_codes
        sos = torch.tensor(
            [[[[2048], [2048], [2048], [2048]]]], dtype=torch.int)
        output_codebook_index = torch.cat((sos, output_codebook_index), 3)

        return output_codebook_index

    def decompress(self, input):
        with torch.no_grad():
            audio_values = self.model.decode(input, [None])[0]
            audio_values = torch.reshape(
                audio_values, (1, -1)).to(torch.float32)
            torchaudio.save(
                "output.wav", audio_values, 32000, format="wav")
