import logging
import wave
import librosa

import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from copy import deepcopy
from pyaudiowpatch import PyAudio, paInt16
from librosa.display import specshow

from PyQt6.QtCore import QThreadPool

from utils.audio_utils.audio_devices import AbstractDevice
from utils.shared_dcs import AudioPipeline, ChromaPipeline, AudioDecomp, ChromaResultSet
from utils.keyidentifier import pitchdistribution as pd
from utils.keyidentifier import classifiers
from utils.chroma_utils.chroma_filters import ChromaFilter
from utils.audio_utils.audio_pipeline import ChromaST, ChromaMT
from utils.qrunnable_utils import GeneralWorker


class ChromaProcessor:
    def __init__(self, adpl: AudioPipeline, chpl: ChromaPipeline):
        self.adpl = adpl
        self.chpl = chpl

        self.threadpool = QThreadPool()

        self.chroma_result = self.get_empty_chroma_result()
        self.fig, self.ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True)
        self.ax.set_facecolor((0, 0, 0))
        self.ax.tick_params(
            axis="both",
            left=False,
            top=False,
            right=False,
            bottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
            labelbottom=False,
        )
        self.fig.tight_layout(pad=0, h_pad=0, w_pad=0)

    def update_adpl(self, adpl: AudioPipeline):
        self.adpl = adpl

    def update_chpl(self, chpl: ChromaPipeline):
        self.chpl = chpl



    def get_empty_chroma_result(self):
        chroma_result = ChromaResultSet(
            chromas=np.empty([12, 0]),
            audio_array=np.empty([0,]),
            key="",
            probability=0,
            bpm=0,
        )
        return chroma_result


    def finish_chromagram(self, chromas, audio_decomp: AudioDecomp):
        len_chroma_result = len(self.chroma_result.chromas[0])

        if len_chroma_result > 1000:
            self.chroma_result = self.get_empty_chroma_result()

        self.chroma_result.chromas = np.concatenate(
            (self.chroma_result.chromas, chromas), axis=1
        )
        self.chroma_result.audio_array = np.concatenate(
            (self.chroma_result.audio_array, audio_decomp.audio_array), axis=0
        )

        self.chroma_result.bpm = self.calculate_bpm(self.chroma_result.audio_array)
        key, probability = self.get_key_probability(self.chroma_result.chromas)
        self.chroma_result.key = key
        self.chroma_result.probability = probability
        specshow(self.chroma_result.chromas, ax=self.ax, sr=22050)

    def save_chromagram(self):
        figure_io = BytesIO()
        self.fig.savefig(figure_io)
        return figure_io

    def calculate_bpm(self, audio_array):
        bpm = librosa.beat.tempo(y=audio_array).flatten()[0]
        bpm = round(bpm, 2)
        return bpm

    def get_result(self):
        return self.chroma_result

    def update_chromagram(self, audio_io: BytesIO):
        self.worker = GeneralWorker(self.update_chromagram_process, audio_io)
        self.threadpool.start(self.worker)

    def update_chromagram_process(self, audio_io: BytesIO):
        audio_decomp = ChromaST(self.adpl).get_audio_decomp(audio_io)
        chromas = audio_decomp.chromas
        p_chromas = self.process_chromas(chromas)
        self.finish_chromagram(p_chromas, audio_decomp)

    def process_chromas(self, chromas):
        chroma_filter = ChromaFilter(chromas)

        if self.chpl.abs_fl_state:
            logging.info("ChromaProcessor:abs_filter")
            chroma_filter.abs_filter()

        if self.chpl.nn_fl_state:
            logging.info("ChromaProcessor:nn_filter")
            chroma_filter.nn_filter()

        if self.chpl.mds_fl_state:
            logging.info("ChromaProcessor:smoothing_filter")
            chroma_filter.smoothing_filter(self.chpl.mds_val)

        if self.chpl.min_clip_fl_state:
            logging.info("ChromaProcessor:clip_filter")
            chroma_filter.clip_filter(self.chpl.min_clip_val)

        p_chromas = chroma_filter.get()
        return p_chromas

    def get_key_probability(self, chromas: np.ndarray) -> tuple[str, float]:
        naive_bayes = classifiers.NaiveBayes()
        dist = pd.PitchDistribution.from_chromagram(chromas)
        key = naive_bayes.get_key(dist)
        probability = float(naive_bayes.get_key_likelihood(key, dist))
        probability = round(probability * 100, 2)
        return key, probability


class AudioRecorder:
    def __init__(self, device: AbstractDevice):
        self.device = device
        self.pyaudio = PyAudio()
        self.frame_buffer = 512
        self.recording = True

    def open_wave_stream(self):
        audio_io = BytesIO()
        wave_io = wave.open(audio_io, "w")

        wave_io.setnchannels(self.device.input_channels)
        wave_io.setsampwidth(self.pyaudio.get_sample_size(paInt16))
        wave_io.setframerate(self.device.sample_rate)
        return wave_io, audio_io

    def get_input_stream(self):
        stream = self.pyaudio.open(
            format=paInt16,
            channels=self.device.input_channels,
            rate=self.device.sample_rate,
            input=True,
            frames_per_buffer=self.frame_buffer,
            input_device_index=self.device.index,
        )
        return stream

    def start_recording(self, chroma_processor: ChromaProcessor):
        stream = self.get_input_stream()
        file_io = BytesIO()

        wave_io, file_io = self.open_wave_stream()
        wave_io2, audio_io = self.open_wave_stream()

        while self.recording is True:
            stream_buffer = stream.read(self.frame_buffer)
            wave_io.writeframes(deepcopy(stream_buffer))
            wave_io2.writeframes(deepcopy(stream_buffer))


            buffer_nbytes = audio_io.getbuffer().nbytes

            if buffer_nbytes > 256_000 or not self.recording:
                # time = (audio_io.getbuffer().nbytes) / (self.device.sample_rate * self.device.input_channels * 16 /8)

                chroma_processor.update_chromagram(audio_io)
                wave_io2, audio_io = self.open_wave_stream()

        chroma_processor.save_chromagram()

        stream.stop_stream()
        stream.close()
        self.pyaudio.terminate()
        wave_io.close()
        return file_io

    def stop_recording(self):
        self.recording = False
