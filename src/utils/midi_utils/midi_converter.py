import numpy as np
from midiutil import MIDIFile
from utils.shared_dcs import MIDINote


class PianoRollMIDI:
    def __init__(self, piano_roll: list[MIDINote], bpm: float):
        self.piano_roll = piano_roll
        self.bpm = bpm

    def get_midi_file(self) -> MIDIFile:
        quarter_note = 60 / self.bpm

        onsets = np.array([p.onset for p in self.piano_roll])
        offsets = np.array([p.offset for p in self.piano_roll])

        onsets = onsets / quarter_note
        offsets = offsets / quarter_note
        durations = offsets - onsets

        midi = MIDIFile(1)
        midi.addTempo(0, 0, self.bpm)

        for i, _ in enumerate(onsets):
            note_value = int(self.piano_roll[i].note_value)
            midi.addNote(0, 0, note_value, onsets[i], durations[i], 100)
        return midi
