import unittest
from unittest.mock import patch

import torch

from src.utils.data import DatasetLoadError, DatasetProtocol


class FakeDataset:
    def __init__(self):
        self.data = {"text": ["tokenized fixture corpus"]}

    def __getitem__(self, key):
        return self.data[key]


class FakeTokenizer:
    model_max_length = 0

    def __call__(self, text, return_tensors, add_special_tokens):
        del text, return_tensors, add_special_tokens
        return type("Encoding", (), {"input_ids": torch.arange(1000).reshape(1, -1)})()


class Phase2Tests(unittest.TestCase):
    def protocol(self, seed_offset=0):
        return DatasetProtocol(
            tokenizer_name="fixture",
            split_config={
                "calibration": {"split": "train", "seq_len": 8, "nsamples": 3, "seed": 10 + seed_offset},
                "validation": {"split": "validation", "seq_len": 8, "nsamples": 3, "seed": 20 + seed_offset},
                "test": {"split": "test", "seq_len": 8, "nsamples": 3, "seed": 30 + seed_offset},
            },
        )

    @patch("src.utils.data.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    @patch("src.utils.data.load_dataset", return_value=FakeDataset())
    def test_protocol_is_reproducible_and_fingerprinted(self, _load, _tokenizer):
        first = self.protocol().load()
        second = self.protocol().load()
        self.assertEqual(first["calibration"].sample_ids, second["calibration"].sample_ids)
        self.assertEqual(first["calibration"].fingerprint, second["calibration"].fingerprint)
        self.assertEqual(first["test"].token_count, 24)
        self.assertTrue(first["validation"].fingerprint)

    @patch("src.utils.data.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    @patch("src.utils.data.load_dataset", return_value=FakeDataset())
    def test_seed_changes_fingerprint(self, _load, _tokenizer):
        first = self.protocol().load()
        changed = self.protocol(seed_offset=1).load()
        self.assertNotEqual(first["calibration"].fingerprint, changed["calibration"].fingerprint)

    @patch("src.utils.data.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    @patch("src.utils.data.load_dataset", return_value=FakeDataset())
    def test_overlapping_protocol_ids_are_rejected(self, _load, _tokenizer):
        protocol = DatasetProtocol(
            tokenizer_name="fixture",
            split_config={
                "calibration": {"split": "train", "seq_len": 8, "nsamples": 2, "seed": 10},
                "validation": {"split": "train", "seq_len": 8, "nsamples": 2, "seed": 10},
                "test": {"split": "test", "seq_len": 8, "nsamples": 2, "seed": 30},
            },
        )
        with self.assertRaises(DatasetLoadError):
            protocol.load()


if __name__ == "__main__":
    unittest.main()