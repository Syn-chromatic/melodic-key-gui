import numpy as np

from scipy.ndimage import median_filter
from librosa.decompose import nn_filter
from typing import Callable


class ChromaFilter:
    def __init__(self, chromas: np.ndarray):
        self.chromas = chromas

    def abs_filter(self):
        self.chromas = np.abs(self.chromas) ** 2
        return self

    def nn_filter(self, aggregate: Callable = np.median, metric: str = "cosine"):
        nnf_chromas = nn_filter(self.chromas, aggregate=aggregate, metric=metric)
        self.chromas = np.minimum(self.chromas, nnf_chromas)
        return self

    def smoothing_filter(self, strength: float):
        strength = int(strength)
        self.chromas = median_filter(self.chromas, size=(1, strength))
        return self

    def clip_filter(self, clip_value: float):
        for chroma_idx in range(len(self.chromas)):
            for idx, chroma_value in enumerate(self.chromas[chroma_idx]):
                if chroma_value < clip_value:
                    self.chromas[chroma_idx][idx] = 0.0
        return self

    def get(self):
        return self.chromas
