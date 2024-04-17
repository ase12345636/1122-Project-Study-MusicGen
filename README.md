# 1122-Project-Study-MusicGen
This is my bachelor project research code. This is reference Simple and Controllable Music Generation.

NOTE:
All comments are based on Musicgen - Melody Model

audiocraft/train -> ./solvers/builders -> ./solvers/musicgen ->

   |-> ./solvers/compression -> ./models/builders > ./models/encodec -> ./models/quantization/qt -> |
   |                                                                                                |
   |                        |-> ./modules/conditioners -> ./modules/chroma --> |                    |
   |                        |                                                  |                    |
-> |-> ./models/builders -> |-> ./modules/codebooks_patterns ----------------> |------------------> | ->
   |                        |                                                  |                    |
   |                        |-> ./models/lm ---------------------------------> |                    |
   |                                                                                                |
   |-> ./solvers/builders -> optimizer ( haven't seen ) ------------------------------------------> |

-> ./solvers/base -> ./solvers/musicgen



cfg :
    Musicgen - Melody Model ( text + melody ) :
        total parameter : 1.5B

        input :
            channels : 1
            sample_rate : 32000

        conditioners:
            self_wav:
                chroma_stem :
                    n_chroma : 12
                    radix2_exp : 14
                    n_eval_wavs : 100
            description :
                t5 :
                    name : t5-base
                    word_dropout : 0.2

        lm model :
            dim : 1536
            num_heads : 24
            num_layers : 48
            n_q : 4
            card : 2048
            modeling: delay

        encodec :
            encodec_large_nq4_s640
            input sample rate : 32khz
            latent code frame rate : 50 frames/s
            rvq :
                n_q : 4
                bins : 2048

        batch_size : 192
        epochs: 500

        optimizer : dadam
        ema : 
            updates: 10
        lr_scheduler : cosine
            warmup: 4000
            lr_min_ratio: 0.0
            cycle_length: 1.0