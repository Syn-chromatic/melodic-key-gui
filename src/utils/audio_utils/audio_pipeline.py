import os
import io
import logging
import librosa
import numpy as np
import soundfile as sf

from io import BytesIO
from pydub import AudioSegment
from copy import deepcopy

from utils.shared_dcs import AudioDecomp, AudioPipeline
from multiprocessing.pool import Pool


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class ChromaMT:
    def __init__(self, adpl: AudioPipeline):
        self.adpl = adpl

    def get_audio_decomp(self, input_io):
        logging.info("ChromaMT:get_audio_decomp")
        pyd_audio_in = AudioSegment.from_file(
            input_io, format="wav", codec="pcm_s16le"
        )
        pyd_io = BytesIO()
        pyd_audio_in.export(pyd_io, format="wav", codec="pcm_s16le")

        y, sr = librosa.load(pyd_io)

        pyd_audio = self.pyd_pipeline(pyd_audio_in)
        y, sr = self.libr_pipeline(pyd_audio)

        if self.adpl.save_out_state:
            sf.write("output.wav", data=y, samplerate=sr)

        chromas = self.compute_chromas(y, sr)
        bpm = self.get_bpm(y)

        audio_decomp = AudioDecomp(
            chromas=chromas,
            audio_array=y,
            sample_rate=sr,
            bpm=bpm,
        )
        return audio_decomp

    def get_bpm(self, audio_array: np.ndarray):
        if self.adpl.calc_bpm_state:
            bpm = librosa.beat.tempo(y=audio_array).flatten()[0]
            bpm = round(bpm, 2)
            return bpm
        return 0

    def low_high_filters(self, audio):
        logging.info("ChromaMT:low_high_filters")
        if self.adpl.lpass_fl_state:
            audio = audio.low_pass_filter(self.adpl.lpass_val)
        if self.adpl.hpass_fl_state:
            audio = audio.high_pass_filter(self.adpl.hpass_val)
        return audio

    def libr_load(self, audio):
        logging.info("ChromaMT:libr_load")
        audioBytes = io.BytesIO()
        audio.export(audioBytes, format="wav", codec="pcm_s16le")
        y, sr = librosa.load(audioBytes)

        y = librosa.util.normalize(S=y) * 0.8
        if self.adpl.inst_fl_state:
            y_harmonic = librosa.effects.harmonic(y=y, margin=1)
            y = y_harmonic

        y_trim, _ = librosa.effects.trim(y=y, frame_length=16, top_db=60, hop_length=1)
        self.libr_fadein(y_trim, sr, duration=0.005)
        self.libr_fadeout(y_trim, sr, duration=0.005)
        return y_trim, sr

    def libr_fadeout(self, audio, sr, duration=3.0):
        logging.info("ChromaMT:libr_fadeout")
        length = int(duration * sr)
        end = audio.shape[0]
        start = end - length
        fade_curve = np.linspace(1.0, 0.0, length)
        audio[start:end] = audio[start:end] * fade_curve

    def libr_fadein(self, audio, sr, duration=3.0):
        logging.info("ChromaMT:libr_fadein")
        end = int(duration * sr)
        start = 0
        fade_curve = np.linspace(0.0, 1.0, end)
        audio[start:end] = audio[start:end] * fade_curve

    def get_segment_ranges(self, n_frames: int) -> list[tuple[int, int]]:
        logging.info("ChromaMT:get_segment_ranges")
        frame_per_div = int(n_frames // self.adpl.core_count)
        num = 0
        segments = []
        for _ in range(self.adpl.core_count):
            segments.append((num, num + frame_per_div))
            num += frame_per_div
        return segments

    def get_bar_count(self, audio, bpm):
        logging.info("ChromaMT:get_bar_count")
        audio_duration = audio.duration_seconds
        bps = bpm / 60
        beat_sec = 1 / bps
        bar_sec = beat_sec * 4
        bars = audio_duration / bar_sec
        return bars

    def get_pyd_slices(self, pyd_audio):
        logging.info("ChromaMT:get_pyd_slices")
        n_frames = int(pyd_audio.frame_count())
        segment_ranges = self.get_segment_ranges(n_frames)
        pyd_slices = []
        for st_slice, en_slice in segment_ranges:
            pyd_slice = pyd_audio.get_sample_slice(
                start_sample=st_slice, end_sample=en_slice
            )
            pyd_slices.append(pyd_slice)
        return pyd_slices

    def pyd_pipeline(self, pyd_audio):
        logging.info("ChromaMT:pyd_pipeline")
        if self.adpl.hpass_fl_state or self.adpl.lpass_fl_state:
            pyd_slices = self.get_pyd_slices(pyd_audio)
            pool = Pool(processes=self.adpl.core_count)
            pyd_slices = pool.map(self.low_high_filters, pyd_slices)
            len_segments = len(pyd_slices)
            pool.close()
            pyd_segment = AudioSegment.empty()
            for idx, chunk in enumerate(pyd_slices):
                logging.info(f"Processing Chunk (PyDub) {idx}/{len_segments}")
                chunk_ms = chunk.duration_seconds * 1000
                if not len(pyd_segment):
                    pyd_segment = chunk
                elif chunk_ms <= 2:
                    pyd_segment = pyd_segment.append(chunk, crossfade=0)
                else:
                    pyd_segment = pyd_segment.append(chunk, crossfade=2)
            return pyd_segment
        return pyd_audio

    def libr_pipeline(self, pyd_audio):
        logging.info("ChromaMT:libr_pipeline")
        pyd_slices = self.get_pyd_slices(pyd_audio)
        pool = Pool(processes=self.adpl.core_count)
        libr_slices = pool.map(self.libr_load, pyd_slices)
        len_slices = len(libr_slices)
        pool.close()

        allArrays = np.array([])
        for idx, (y, sr) in enumerate(libr_slices):
            logging.info(f"Processing Chunk (Librosa) - {idx}/{len_slices}")
            allArrays = np.concatenate([allArrays, y])
        y, sr = allArrays, libr_slices[0][1]
        return y, sr

    def compute_chromas(self, y, sr):
        logging.info("ChromaMT:compute_chromas")
        chromas_cqt = librosa.feature.chroma_cqt(
            y=y, sr=sr, n_chroma=12, threshold=5, bins_per_octave=36
        )
        return chromas_cqt


class ChromaST:
    def __init__(self, adpl: AudioPipeline):
        self.adpl = adpl

    def get_audio_decomp(self, input_io: BytesIO):
        logging.info("ChromaST:get_audio_decomp")
        pyd_audio_in = AudioSegment.from_file(input_io, format="wav", codec="pcm_s16le")

        pyd_audio = self.pyd_pipeline(pyd_audio_in)
        y, sr = self.libr_pipeline(pyd_audio)

        if self.adpl.save_out_state:
            sf.write("output.wav", data=y, samplerate=sr)

        chromas = self.compute_chromas(y, sr)
        bpm = self.get_bpm(y)

        audio_decomp = AudioDecomp(
            chromas=chromas,
            audio_array=y,
            sample_rate=sr,
            bpm=bpm,
        )
        return audio_decomp

    def get_bpm(self, audio_array: np.ndarray):
        if self.adpl.calc_bpm_state:
            bpm = librosa.beat.tempo(y=audio_array).flatten()[0]
            bpm = round(bpm, 2)
            return bpm
        return 0

    def low_high_filters(self, pyd_audio):
        logging.info("ChromaST:low_high_filters")
        if self.adpl.lpass_fl_state:
            pyd_audio = pyd_audio.low_pass_filter(self.adpl.lpass_val)
        if self.adpl.hpass_fl_state:
            pyd_audio = pyd_audio.high_pass_filter(self.adpl.hpass_val)
        return pyd_audio

    def libr_harmonic(self, pyd_audio) -> tuple[np.ndarray, float]:
        logging.info("ChromaST:libr_harmonic")
        audio_io = BytesIO()
        pyd_audio.export(audio_io, format="wav", codec="pcm_s16le")

        y, sr = librosa.load(audio_io)
        if self.adpl.inst_fl_state:
            y = librosa.effects.harmonic(y=y, margin=1)

        y = librosa.util.normalize(S=y)
        return y, sr

    def pyd_pipeline(self, pyd_audio):
        logging.info("ChromaST:pyd_pipeline")
        pyd_audio = self.low_high_filters(pyd_audio)
        return pyd_audio

    def libr_pipeline(self, pyd_audio) -> tuple[np.ndarray, float]:
        logging.info("ChromaST:libr_pipeline")
        y, sr = self.libr_harmonic(pyd_audio)
        return y, sr

    def compute_chromas(self, y, sr) -> np.ndarray:
        logging.info("ChromaST:compute_chromas")
        chromas_cqt = librosa.feature.chroma_cqt(
            y=y, sr=sr, n_chroma=12, threshold=5, bins_per_octave=36
        )
        return chromas_cqt
