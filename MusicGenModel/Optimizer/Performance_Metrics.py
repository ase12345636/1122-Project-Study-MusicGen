import os
from pathlib import Path

from frechet_audio_distance import FrechetAudioDistance


class Performance_Metrics:
    '''
    Class for handle FAD score
    '''

    def __init__(self):

        # Initialize
        super().__init__()
        self.frechet = FrechetAudioDistance(model_name="vggish",
                                            use_pca=False,
                                            use_activation=False,
                                            verbose=False)

    def FAD(self, background_path, eval_path):

        # Delete unnecessary folder
        if os.path.isdir(Path(background_path)/".ipynb_checkpoints"):
            (Path(background_path)/".ipynb_checkpoints").rmdir()

        if os.path.isdir(Path(eval_path)/".ipynb_checkpoints"):
            (Path(eval_path)/".ipynb_checkpoints").rmdir()

        # Compute FAD score with background audio and evaluate audio
        fad_score = self.frechet.score(background_path,
                                       eval_path,
                                       dtype="float32")

        return fad_score
