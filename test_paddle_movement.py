"""Regression checks for pitch-controlled paddle travel at every menu size."""

import unittest

from main import pitch_position


class PaddleMovementTests(unittest.TestCase):
    def test_pitch_reaches_both_edges_at_every_size(self):
        neutral = 440
        for height in (720, 1080, 1440):
            for percent in range(50, 201, 10):
                paddle_h = round(height * .23 * percent / 100)
                for semitones, expected_top in ((7, 0), (-7, height - paddle_h)):
                    with self.subTest(height=height, percent=percent, semitones=semitones):
                        center = pitch_position(neutral * 2 ** (semitones / 12), neutral, height, paddle_h)
                        self.assertEqual(round(center - paddle_h / 2), expected_top)

    def test_neutral_stays_centered_after_calibration(self):
        for neutral in (110, 220, 440):
            self.assertEqual(pitch_position(neutral, neutral, 1080, 124), 540)

    def test_extreme_pitches_stay_within_court(self):
        self.assertEqual(pitch_position(1760, 440, 1080, 124), 62)
        self.assertEqual(pitch_position(110, 440, 1080, 124), 1018)


if __name__ == '__main__':
    unittest.main()
