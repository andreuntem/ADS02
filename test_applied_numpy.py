import unittest
import numpy as np
from applied_numpy import build_sequences

class TestBuildSequences(unittest.TestCase):

    def test_seq_1_odd(self):
        min_value = 1
        max_value = 5
        sequence_number = 1
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(calc.tolist(), [1, 3, 5])

    def test_seq_1_even(self):
        min_value = 2
        max_value = 10
        sequence_number = 1
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(calc.tolist(), [3, 5, 7, 9])

    def test_seq_2_in(self):
        min_value = 30
        max_value = 50
        sequence_number = 2
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(sorted(calc.tolist()), [30, 35, 40, 45, 50])

    def test_seq_2_out(self):
        min_value = 29
        max_value = 41
        sequence_number = 2
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(sorted(calc.tolist()), [30, 35, 40])

    def test_seq_3_in(self):
        min_value = 4
        max_value = 64
        sequence_number = 3
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(calc.tolist(), [4, 8, 16, 32, 64])

    def test_seq_3_out(self):
        min_value = 60
        max_value = 1025
        sequence_number = 3
        calc = build_sequences(min_value, max_value, sequence_number)
        self.assertEqual(calc.tolist(), [64, 128, 256, 512, 1024])
