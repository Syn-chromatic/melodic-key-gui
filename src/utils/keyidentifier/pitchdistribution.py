#!/usr/bin/env python
"""
Class PitchDistribution represents proportion of musical sample made up of each note A, A#, B, ..., G, G#.
"""

from utils.keyidentifier import audioprocessing as ap
import numpy as np
from decimal import Decimal

NUM_NOTES = 12
NOTES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
INTERVALS = ['P1', 'm2', 'M2', 'm3', 'M3', 'P4', 'd5', 'P5', 'm6', 'M6', 'm7', 'M7']
SCALES = ['major', 'minor']
# 'Typical' pitch distributions [A, A#, B, ..., G#] for major, minor scales with tonic A,
# adapted for use with other tonal centers by rotating.
# MAJOR_SCALE_PROFILE = [0.16, 0.03, 0.09, 0.03, 0.13, 0.10, 0.06, 0.14, 0.03, 0.11, 0.03, 0.09]
# MINOR_SCALE_PROFILE = [0.16, 0.03, 0.09, 0.13, 0.03, 0.10, 0.06, 0.14, 0.11, 0.03, 0.09, 0.03]




# MAJOR_SCALE_PROFILE = [8.04, 0.62, 10.57, 16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80]
# MINOR_SCALE_PROFILE = [1.53, 0.92, 10.21, 18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49]



MAJOR_SCALE_PROFILE = [0.0804, 0.006200000000000001, 0.10570000000000002, 0.16800000000000004, 0.008600000000000002, 0.1295, 0.014100000000000001, 0.13490000000000002, 0.11930000000000002, 0.012500000000000002, 0.20280000000000004, 0.018000000000000002]
MINOR_SCALE_PROFILE = [0.015300000000000003, 0.009200000000000002, 0.10210000000000002, 0.18160000000000004, 0.006900000000000001, 0.12990000000000002, 0.13340000000000002, 0.010700000000000003, 0.11150000000000002, 0.013800000000000002, 0.21070000000000003, 0.07490000000000001]




def skip_interval(root, interval):
    """
    Returns note which is INTERVAL distance away from starting note ROOT.
    """
    assert root in NOTES, "Invalid note"
    assert interval in INTERVALS, "Invalid interval"
    starting_position = NOTES.index(root)
    distance = INTERVALS.index(interval)
    return NOTES[(starting_position + distance) % NUM_NOTES]


class Key(object):
    """
    Key centered at tonal center TONIC with scale SCALE
    """
    scale_profiles = {'major': MAJOR_SCALE_PROFILE, 'minor': MINOR_SCALE_PROFILE}

    def __init__(self, tonic, scale):
        assert tonic in NOTES, "Tonal center of key must belong to NOTES"
        assert scale in SCALES, "Scale for key must belong to SCALES"
        self.tonic = tonic
        self.scale = scale

    def __str__(self):
        return ' '.join((self.tonic, self.scale))

    def __repr__(self):
        return ''.join(("Key('", self.tonic, "', '", self.scale, "')"))

    def __eq__(self, other):
        if type(other) != Key:
            return False
        return self.tonic == other.tonic and self.scale == other.scale

    def get_tonic(self):
        return self.tonic

    def get_scale(self):
        return self.scale

    def get_key_profile(self):
        """
        Return typical PitchDistribution for Key
        """
        scale_profile = Key.scale_profiles[self.scale]
        key_profile = PitchDistribution()
        for i in range(NUM_NOTES):
            current_note = skip_interval(self.tonic, INTERVALS[i])
            val = scale_profile[i]
            key_profile.set_val(current_note, val)
        return key_profile


class PitchDistribution(object):
    """
    Distribution over pitch classes A, A#, ..., G, G# in the form of a map NOTES --> [0,1]
    """
    def __init__(self, values_array=None):
        """
        Initializes empty distribution.
        """
        self.distribution = {}
        if values_array:
            assert len(values_array) == NUM_NOTES, "Distribution must have %d notes, %d provided" % (NUM_NOTES, len(values_array))
            for i in range(NUM_NOTES):
                note = NOTES[i]
                val = values_array[i]
                self.set_val(note, val)
            self.normalize()

    @classmethod
    def from_file(cls, filename):
        """
        Given path FILENAME to audio file, return its PitchDistribution
        """
        def chromagram_index_to_note(i):
            """
            Given row index in librosa chromagram, returns note it represents
            """
            return skip_interval('C', INTERVALS[i])

        C = ap.chromagram_from_file(filename)

        # Pick out only most prominent note in each time interval
        single_note_reduction = C.argmax(axis=0)

        dist = PitchDistribution()
        for i in np.nditer(single_note_reduction):
            note = chromagram_index_to_note(i)
            dist.increment_val(note)
        dist.normalize()
        return dist


    @classmethod
    def from_array(cls, y, sr):
        """
        Given path FILENAME to audio file, return its PitchDistribution
        """
        def chromagram_index_to_note(i):
            """
            Given row index in librosa chromagram, returns note it represents
            """
            return skip_interval('C', INTERVALS[i])

        C = ap.chromagram_from_array(y, sr)

        # Pick out only most prominent note in each time interval
        single_note_reduction = C.argmax(axis=0)

        dist = PitchDistribution()
        for i in np.nditer(single_note_reduction):
            note = chromagram_index_to_note(i)
            dist.increment_val(note)
        dist.normalize()
        return dist

    @classmethod
    def from_chromagram(cls, C):
        """
        Given path FILENAME to audio file, return its PitchDistribution
        """
        def chromagram_index_to_note(i):
            """
            Given row index in librosa chromagram, returns note it represents
            """
            return skip_interval('C', INTERVALS[i])

        # Pick out only most prominent note in each time interval
        single_note_reduction = C.argmax(axis=0)


        # dist = PitchDistribution()
        # iter = 0
        # for i in C:
        #     for ii in i:
        #         if ii > 0.001:
        #             note = chromagram_index_to_note(iter)
        #             dist.increment_val(note)
        #     iter += 1
        # dist.normalize()

        dist = PitchDistribution()
        for i in np.nditer(single_note_reduction):
            note = chromagram_index_to_note(i)
            dist.increment_val(note)
        dist.normalize()

        return dist

    def __str__(self):
        return str([(note, self.get_val(note)) for note in NOTES])

    def to_array(self):
        return [self.get_val(note) for note in NOTES]

    def set_val(self, note, val):
        self.distribution[note] = Decimal(val)

    def get_val(self, note):
        if note in self.distribution:
            return self.distribution[note]
        return Decimal(0.0)

    def increment_val(self, note):
        """
        Increments value of note NOTE in a distribution
        """
        self.set_val(note, self.get_val(note) + 1)

    def normalize(self):
        """
        Normalize distribution so that all entries sum to 1
        """
        distribution_sum = sum(self.distribution.values())
        if distribution_sum != 0:
            for note in NOTES:
                val = self.get_val(note)
                self.set_val(note, Decimal(val) / Decimal(distribution_sum))