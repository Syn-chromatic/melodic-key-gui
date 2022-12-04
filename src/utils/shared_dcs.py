import numpy as np
from dataclasses import dataclass


@dataclass
class OnsetMIDINote:
    note_value: int
    note_name: str
    onset: float


@dataclass
class MIDINote:
    note_value: int
    note_name: str
    onset: float
    offset: float


@dataclass
class AudioDecomp:
    chromas: np.ndarray
    audio_array: np.ndarray
    sample_rate: float
    bpm: float


@dataclass
class ChromaResultSet:
    chromas: np.ndarray
    audio_array: np.ndarray
    key: str
    probability: float
    bpm: float


@dataclass
class AudioPipeline:
    hpass_fl_state: bool
    hpass_val: int
    lpass_fl_state: bool
    lpass_val: int
    inst_fl_state: bool
    save_out_state: bool
    calc_bpm_state: bool
    core_count: int


@dataclass
class ChromaPipeline:
    abs_fl_state: bool
    nn_fl_state: bool
    mds_fl_state: bool
    mds_val: float
    min_clip_fl_state: bool
    min_clip_val: float
