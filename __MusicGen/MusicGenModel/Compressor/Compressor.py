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

    def compress(self, file_path: list[str], mode: str = "Parallel"):
        '''
        Input : file_path
            A batch of audio file path

            shape : [audio file path]


    ->  Codebook Embedding : codebook_index
            A batch of token id

            shape : [batch, batch size, codebook, length]

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

            # Proccess parallel pattern
            if mode == "Parallel":

                # Add SOS token
                sos = torch.IntTensor(
                    [[[[2048], [2048], [2048], [2048]]]])

                # Concat encodec embedding with SOS token
                codebook_index = torch.cat(
                    (sos, codebook_index[:, :, :, :self.max_length]), 3)

            elif mode == "Delay":

                # Add SOS token and SP token
                new_codebook_index = None

                #  Concat Concat encodec embedding with SP token
                for codebook in range(4):

                    # Proccess with each codebook embedding
                    # Add SP token in front of codebook embedding
                    temp_codebook_index = codebook_index[:, :, codebook, :]
                    for i in range(codebook):
                        temp_codebook_index = torch.cat(
                            (torch.IntTensor([[[2049]]]), temp_codebook_index), 2)

                    # Add SP token behind codebook embedding
                    for i in range(3-codebook):
                        temp_codebook_index = torch.cat(
                            (temp_codebook_index, torch.IntTensor([[[2049]]])), 2)

                    if new_codebook_index == None:
                        new_codebook_index = temp_codebook_index

                    else:
                        new_codebook_index = torch.cat(
                            (new_codebook_index, temp_codebook_index), 1)

                # Add SOS token in front of codebook embedding
                codebook_index = torch.reshape(
                    new_codebook_index, (1, 1, 4, -1))

                sos = torch.IntTensor(
                    [[[[2048], [2048], [2048], [2048]]]])
                codebook_index = torch.cat(
                    (sos, codebook_index[:, :, :, :self.max_length+3]), 3)

            tgt_input = torch.cat((tgt_input, codebook_index), 1)

        # Return
        return tgt_input

    def decompress(self, input, file_path: str):
        audio_values = self.encodec.decode(input, [None])[0]
        audio_values = torch.reshape(
            audio_values, (1, -1)).to(torch.float32)
        torchaudio.save(
            file_path, audio_values, 32000, format="wav")
