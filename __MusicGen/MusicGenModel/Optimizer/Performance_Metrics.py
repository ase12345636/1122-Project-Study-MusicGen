from frechet_audio_distance import FrechetAudioDistance


class Performance_Metrics:
    '''
    Class for handle text condition
    '''

    def __init__(self):
        # Initialize
        super().__init__()
        self.frechet = FrechetAudioDistance(model_name="vggish",
                                            use_pca=False,
                                            use_activation=False,
                                            verbose=False)

    def FAD(self, background_path: str, eval_path: str):
        fad_score = self.frechet.score(background_path,
                                       eval_path,
                                       dtype="float32")

        return fad_score
