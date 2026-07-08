from __future__ import annotations

import unittest

from src.data.split import leave_two_out


class LeaveTwoOutSplitTest(unittest.TestCase):
    def test_leave_two_out(self) -> None:
        split = leave_two_out([2, 3, 4, 5, 6], min_train_length=3)
        self.assertEqual(split.train, [2, 3, 4])
        self.assertEqual(split.validation, 5)
        self.assertEqual(split.test, 6)

    def test_short_sequence_raises(self) -> None:
        with self.assertRaises(ValueError):
            leave_two_out([2, 3, 4], min_train_length=2)


if __name__ == "__main__":
    unittest.main()

