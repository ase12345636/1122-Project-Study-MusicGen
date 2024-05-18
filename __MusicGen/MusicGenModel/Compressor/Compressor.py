import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor
from Config.Config import device


class Compressor():
    '''
    Class for compress/ decompress audio
    '''

    def __init__(self, src_sample_rate: int = 48000, tgt_sample_rate: int = 32000, max_length: int = 250):
        # Initialize
        super().__init__()
        self.max_length = max_length

        self.resampler = torchaudio.transforms.Resample(
            src_sample_rate, tgt_sample_rate)
        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_32khz")
        self.encodec = EncodecModel.from_pretrained("facebook/encodec_32khz")

    def compress(self, file_path: list[str]):
        '''
        Input : file_path
            A batch of audio file path

            shape : [audio file path]


    ->  Token : text_condition_ids
            A batch of token id

            shape : [batch size, length]

    ->  Output : tgt_input
            A batch of tgt output

            shape : [batch, batch size, length, codebook]
        '''

        # Get tgt input
        tgt_input = torch.IntTensor([])
        for audio_path in file_path[0]:
            audio, _ = torchaudio.load(audio_path)

            # Convert audio from stereo to mono
            audio = torch.mean(audio, dim=0)

            # Convert audio sample rate from 48khz to 32khz
            audio = self.resampler(audio)

            # Compress audio
            audio = self.processor(audio, return_tensors="pt")
            codebook_index = self.encodec.encode(
                audio["input_values"], audio["padding_mask"]).audio_codes

            # Add SOS token
            sos = torch.tensor(
                [[[[2048], [2048], [2048], [2048]]]], dtype=torch.int)
            codebook_index = torch.cat(
                (sos, codebook_index[:, :, :, :self.max_length]), 3)

            tgt_input = torch.cat((tgt_input, codebook_index), 1)

        # Return
        return tgt_input

    def decompress(self, input, file_path: str):
        audio_values = self.encodec.decode(input, [None])[0]
        audio_values = torch.reshape(
            audio_values, (1, -1)).to(torch.float32)
        torchaudio.save(
            file_path, audio_values, 32000, format="wav")
