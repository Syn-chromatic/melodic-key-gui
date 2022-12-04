import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from librosa.display import specshow

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QWidget, QFileDialog
from ui.chromaconfig_ui import Ui_Dialog

from utils.keyidentifier import pitchdistribution as pd
from utils.keyidentifier import classifiers
from utils.midi_utils.midi_converter import PianoRollMIDI
from utils.chroma_utils.chroma_pianoroll import ChromaPianoRoll
from utils.shared_dcs import AudioDecomp
from utils.chroma_utils.chroma_filters import ChromaFilter

from utils.qrunnable_utils import GeneralWorker


class ChromaDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()

    def show_window(self, audio_decomp: AudioDecomp):
        self.show()

        self.keys = ""
        self.bpm = 0
        self.sr = 0

        self.audio_decomp = audio_decomp
        self.ui.updateChroma.setEnabled(False)
        self.ui.showChroma.clicked.connect(self.show_chromagram)
        self.ui.updateChroma.clicked.connect(self.update_chromagram)
        self.ui.convertMidi.clicked.connect(self.save_midi)

    def show_chromagram(self):
        with plt.ion():
            self.fig, self.ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True)
            self.update_chromagram()
            self.ui.updateChroma.setEnabled(True)

    def enable_buttons(self):
        self.ui.showChroma.setEnabled(True)
        self.ui.updateChroma.setEnabled(True)
        self.ui.convertMidi.setEnabled(True)

    def disable_buttons(self):
        self.ui.showChroma.setEnabled(False)
        self.ui.updateChroma.setEnabled(False)
        self.ui.convertMidi.setEnabled(False)

    def finish_chromagram(self, chromas):
        self.enable_buttons()
        self.ax.set(title=f"Keys: {self.keys} - BPM: {self.bpm}")
        specshow(chromas, y_axis="chroma", x_axis="time", ax=self.ax, sr=int(self.sr))
        plt.pause(0.0001)

    def update_chromagram(self):
        self.disable_buttons()
        worker = GeneralWorker(self.update_chromagram_process)
        worker.signals.output.connect(self.finish_chromagram)
        self.threadpool.start(worker)

    def update_chromagram_process(self):
        chromas = self.process_chromas()
        self.keys = self.get_key_probability(chromas)
        self.bpm = self.audio_decomp.bpm
        self.sr = self.audio_decomp.sample_rate
        return chromas

    def process_chromas(self):
        chromas = deepcopy(self.audio_decomp.chromas)
        chroma_filter = ChromaFilter(chromas)

        if self.ui.absoluteFilter.isChecked():
            chroma_filter.abs_filter()

        if self.ui.nnFilter.isChecked():
            chroma_filter.nn_filter()

        if self.ui.mSmoothingFilter.isChecked():
            smoothing_value = int(self.ui.mSmoothingValue.text())
            chroma_filter.smoothing_filter(smoothing_value)

        if self.ui.minClipFilter.isChecked():
            min_clip = float(self.ui.minClipValue.text())
            chroma_filter.clip_filter(min_clip)

        p_chromas = chroma_filter.get()
        return p_chromas

    def save_midi(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save File", filter="MIDI File (*.mid)"
        )
        filepath = Path(filepath)

        self.disable_buttons()
        worker = GeneralWorker(self.chroma_to_midi, filepath)
        worker.signals.output.connect(self.enable_buttons)
        self.threadpool.start(worker)

    def chroma_to_midi(self, filepath: Path):
        chromas = self.process_chromas()
        sr = self.audio_decomp.sample_rate
        bpm = self.audio_decomp.bpm

        piano_roll = ChromaPianoRoll(chromas, int(sr)).get_piano_roll()
        midi_file = PianoRollMIDI(piano_roll, bpm).get_midi_file()

        with open(filepath, "wb") as f:
            midi_file.writeFile(f)

    def get_key_probability(self, chromas: np.ndarray):
        naive_bayes = classifiers.NaiveBayes()
        dist = pd.PitchDistribution.from_chromagram(chromas)
        keys = []
        key_probs = {}
        for note in pd.NOTES:
            for scale in pd.SCALES:
                keys.append(f"{note} {scale}")

        for key in keys:
            key_likelihood = naive_bayes.get_key_likelihood(key, dist)
            key_probs.update({key: key_likelihood})

        key_probs = dict(sorted(key_probs.items(), key=lambda x: x[1], reverse=True))
        top_key_probs = list(key_probs.keys())[:3]
        return top_key_probs
