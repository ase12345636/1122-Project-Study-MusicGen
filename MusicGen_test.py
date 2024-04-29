# Colab 上可以成功執行
# pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch

model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
model.set_generation_params(duration=8)

# 產生5首音樂出來
res = model.generate([
    'crazy EDM, heavy bang', 
    'classic reggae track with an electronic guitar solo',
    'lofi slow bpm electro chill with organic samples',
    'rock with saturated guitars, a heavy bass line and crazy drum break and fills.',
    'earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves',
], 
    progress=True)
# 將音樂show出來
display_audio(res, 32000)