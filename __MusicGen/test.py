import torch
import torchaudio
import torchaudio.transforms as T
from transformers import EncodecModel, AutoProcessor, EncodecFeatureExtractor

processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
model = EncodecModel.from_pretrained("facebook/encodec_32khz")
model_codebook = EncodecFeatureExtractor.from_pretrained(
    "facebook/encodec_32khz")

metasata, _ = torchaudio.load(
    "D:\\Project\\MachineLearning\\__MusicGen\Dataset\\Music_2.wav")
metasata = (metasata[0]+metasata[1])/2
# print(type(metasata))
# print(metasata)
# print(metasata.shape)

resampler = T.Resample(
    48000, 32000)
metasata = resampler(metasata)
input = processor(
    metasata, sampling_rate=32000, return_tensors="pt")

# print(type(inputs["input_values"]))
# print(inputs["input_values"])
# print(inputs["input_values"].shape)

# encoder_outputs = model.encode(
#     inputs["input_values"], inputs["padding_mask"])
# print(type(encoder_outputs.audio_codes))
# print(encoder_outputs.audio_codes)
# print(encoder_outputs.audio_codes.shape)

output_codebook_index = model.encode(
    input["input_values"], input["padding_mask"]).audio_codes

output_codebook_value = torch.permute(model.quantizer.decode(
    output_codebook_index[0]), (0, 2, 1))
sos = torch.zeros([4, 1, 128], dtype=torch.float)
output_codebook_value = torch.cat((sos, output_codebook_value), 1)

print(output_codebook_value.shape)
