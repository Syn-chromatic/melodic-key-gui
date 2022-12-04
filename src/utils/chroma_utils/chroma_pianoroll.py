import librosa

import numpy as np
from utils.shared_dcs import MIDINote, OnsetMIDINote


class ChromaPRBase:
    def __init__(self, chromas: np.ndarray, sample_rate: int):
        self._chromas = chromas
        self._sample_rate = sample_rate
        self._hop_length = 512
        self._note_offset = 48

    def _get_hop_time(self) -> float:
        return self._hop_length / self._sample_rate

    def _get_len_chroma_duration(self) -> int:
        return len(self._chromas[0])

    def _get_len_chromas(self):
        return len(self._chromas)


class ChromaPianoRoll(ChromaPRBase):
    def __init__(self, chromas: np.ndarray, sample_rate: int):
        super().__init__(chromas, sample_rate)

    def get_piano_roll(self) -> list[MIDINote]:
        piano_roll: list[MIDINote] = []
        note_dict: dict[int, OnsetMIDINote] = {}
        expired_notes = []

        len_chromas = self._get_len_chromas()
        len_chroma_duration = self._get_len_chroma_duration()
        hop_time = self._get_hop_time()

        for chroma_pos in range(len_chroma_duration):
            note_values = []
            for chroma_idx in range(len_chromas):
                chroma_value = self._chromas[chroma_idx][chroma_pos]
                if chroma_value > 0.5:
                    note_values.append(chroma_idx)

            for note_value in note_values:
                if note_value not in note_dict:
                    note_name = str(librosa.midi_to_note(note_value))
                    note_onset = chroma_pos * hop_time
                    onset_midi = OnsetMIDINote(
                        note_value=note_value,
                        note_name=note_name,
                        onset=note_onset,
                    )
                    note_dict.update({note_value: onset_midi})

            for note, onset_midi in note_dict.items():
                if note not in note_values:
                    note_value = note + self._note_offset
                    note_name = str(librosa.midi_to_note(note_value))
                    note_onset = onset_midi.onset
                    note_offset = chroma_pos * hop_time
                    midi_note = MIDINote(
                        note_value=note_value,
                        note_name=note_name,
                        onset=note_onset,
                        offset=note_offset,
                    )
                    piano_roll.append(midi_note)
                    expired_notes.append(note)

            for exp_note in expired_notes:
                note_dict.pop(exp_note)
            expired_notes = []
        return piano_roll
