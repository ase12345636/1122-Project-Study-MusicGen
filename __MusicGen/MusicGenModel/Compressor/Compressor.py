import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor
from Config.Config import device, SOS_token, SP_token, melody_condition_max_length


class Compressor():
    '''
    Class for compress/ decompress audio
    '''

    def __init__(self, src_sample_rate: int = 48000, tgt_sample_rate: int = 32000, max_length: int = 250,
                 mode: str = "Delay"):
        super().__init__()

        # Initialize
        self.max_length = max_length

        self.resampler = torchaudio.transforms.Resample(
            src_sample_rate, tgt_sample_rate)
        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_32khz")
        self.encodec = EncodecModel.from_pretrained(
            "facebook/encodec_32khz")

        self.mode = mode

    def resample_audio(self, src_file_path: str, tgt_file_path: str):

        # Get tgt input
        audio, _ = torchaudio.load(src_file_path)

        # Convert audio from stereo to mono
        audio = torch.mean(audio, dim=0)

        # Convert audio sample rate from 48khz to 32khz
        audio = torch.reshape(self.resampler(audio)[:32000*5], (1, -1))

        torchaudio.save(
            tgt_file_path, audio, 32000, format="wav")

    def compress(self, file_path: list[str]):
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

            '''
            Parallel Pattern Mode :

                SOS Token : SOS_token

                Output Content :
                [[SOS_token, x, x, x, ......],
                 [SOS_token, x, x, x, ......],
                 [SOS_token, x, x, x, ......],
                 [SOS_token, x, x, x, ......]]
            '''
            # Proccess parallel pattern
            if self.mode == "Parallel":

                # Add SOS token
                sos = torch.IntTensor(
                    [[[[SOS_token], [SOS_token], [SOS_token], [SOS_token]]]])

                # Concat encodec embedding with SOS token
                codebook_index = torch.cat(
                    (sos, codebook_index[:, :, :, :self.max_length]), 3)

            '''
            Parallel Delay Mode :

                SOS Token : SOS_token
                SP Token : SP_token

                Output Content :
                [[SOS_token, x,               x,        x, ......, SP_token, SP_token, SP_token],
                 [SOS_token, SP_token,        x,        x, ......,        x, SP_token, SP_token],
                 [SOS_token, SP_token, SP_token,        x, ......,        x,        x, SP_token],
                 [SOS_token, SP_token, SP_token, SP_token, ......,        x,        x,        x]]
            '''
            # Proccess delay pattern
            if self.mode == "Delay":

                # Add SOS token and SP token
                new_codebook_index = None

                #  Concat Concat encodec embedding with SP token
                for codebook in range(4):

                    # Proccess with each codebook embedding
                    # Add SP token in front of codebook embedding
                    temp_codebook_index = codebook_index[:, :,
                                                         codebook, :self.max_length]

                    for i in range(codebook):
                        temp_codebook_index = torch.cat(
                            (torch.IntTensor([[[SP_token]]]), temp_codebook_index), 2)

                    # Add SP token behind codebook embedding
                    for i in range(3-codebook):
                        temp_codebook_index = torch.cat(
                            (temp_codebook_index, torch.IntTensor([[[SP_token]]])), 2)

                    if new_codebook_index == None:
                        new_codebook_index = temp_codebook_index

                    else:
                        new_codebook_index = torch.cat(
                            (new_codebook_index, temp_codebook_index), 1)

                # Add SOS token in front of codebook embedding
                codebook_index = torch.reshape(
                    new_codebook_index, (1, 1, 4, -1))

                sos = torch.IntTensor(
                    [[[[SOS_token], [SOS_token], [SOS_token], [SOS_token]]]])
                codebook_index = torch.cat(
                    (sos, codebook_index[:, :, :, :self.max_length+3]), 3)

            tgt_input = torch.cat((tgt_input, codebook_index), 1)

        # Return
        return tgt_input

    def decompress(self, input, file_path: str):
        self.encodec = self.encodec.to(device)

        if self.mode == "Parallel":
            audio_values = self.encodec.decode(input, [None])[0]
            audio_values = torch.reshape(
                audio_values, (1, -1)).to(torch.float32).cpu()

            torchaudio.save(
                file_path, audio_values, 32000, format="wav")

        if self.mode == "Delay":

            new_input = None
            for codebook in range(4):
                # Skip SP Token
                temp_input = input[0, 0, codebook, codebook:]
                temp_input = temp_input[:melody_condition_max_length]

                if new_input == None:
                    new_input = torch.reshape(
                        temp_input, (1, 1, 1, melody_condition_max_length))

                else:
                    new_input = torch.cat((new_input, torch.reshape(
                        temp_input, (1, 1, 1, melody_condition_max_length))), 2)

            audio_values = self.encodec.decode(new_input, [None])[0]
            audio_values = torch.reshape(
                audio_values, (1, -1)).to(torch.float32).cpu()

            torchaudio.save(
                file_path, audio_values, 32000, format="wav")
