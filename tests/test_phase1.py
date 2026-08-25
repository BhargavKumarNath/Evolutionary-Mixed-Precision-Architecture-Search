import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.core.configuration import ConfigurationError, configuration_hash, validate_config
from src.core.evaluator import Evaluator, FitnessMetrics
from src.core.search_space import Genome, MixedPrecisionSearchSpace
from src.core.validator import Validator
from src.engine.ga import GAEngine
from src.utils.data import DatasetLoadError, get_calibration_dataset
from src.utils.manifest import build_manifest, write_manifest
from src.utils.reproducibility import seed_everything


class FixedEvaluator(Evaluator):
    def evaluate(self, genome: Genome) -> FitnessMetrics:
        average = sum(genome.genes) / len(genome.genes)
        return FitnessMetrics(
            mean_loss=average,
            peak_memory_mb=sum(genome.genes),
            latency_ms=16.0 - average,
        )


class FakeModel:
    def eval(self):
        return self


class FakeWrapper:
    model = FakeModel()

    def forward_pass_check(self, batch: torch.Tensor) -> float:
        return 2.0 if batch.numel() == 2 else 4.0


class Phase1Tests(unittest.TestCase):
    def test_validator_keeps_loss_and_latency_separate(self):
        validator = Validator.__new__(Validator)
        validator.wrapper = FakeWrapper()
        validator.dataset = [torch.ones(1, 2), torch.ones(1, 4)]

        result = validator.validate()

        self.assertAlmostEqual(result.mean_loss, 10.0 / 3.0)
        self.assertAlmostEqual(result.perplexity, np.exp(10.0 / 3.0))
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertGreater(result.throughput_tokens_per_sec, 0.0)
        self.assertNotEqual(result.mean_loss, result.latency_ms)

    def test_dataset_failure_is_explicit_by_default(self):
        with patch("src.utils.data.load_dataset", side_effect=OSError("offline")), self.assertRaises(
            DatasetLoadError
        ):
            get_calibration_dataset("unused", seq_len=4, nsamples=1)

    def test_synthetic_data_requires_explicit_opt_in(self):
        with patch("src.utils.data.load_dataset", side_effect=OSError("offline")):
            samples = get_calibration_dataset(
                "unused", seq_len=4, nsamples=2, seed=7, allow_synthetic_data=True
            )
        self.assertEqual([tuple(sample.shape) for sample in samples], [(1, 4), (1, 4)])

    def test_seed_everything_reproduces_supported_rngs(self):
        seed_everything(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        seed_everything(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_search_space_enforces_average_bitwidth(self):
        space = MixedPrecisionSearchSpace(2, [2, 4, 8], min_avg_bitwidth=4.0)
        self.assertFalse(space.validate(Genome(genes=[2, 2])))
        self.assertTrue(space.validate(Genome(genes=[2, 8])))
        self.assertFalse(space.validate(Genome(genes=[2])))

    def test_configuration_validation_rejects_invalid_algorithm(self):
        config = {
            "seed": 42,
            "target_device": "cpu",
            "search_space": {"num_layers": 2, "choices": [4, 8]},
            "algorithm": {
                "pop_size": 4,
                "n_generations": 2,
                "tournament_size": 2,
                "checkpoint_freq": 1,
                "crossover_rate": 0.9,
                "mutation_rate": 0.1,
            },
            "evaluator": {"type": "proxy"},
        }
        self.assertEqual(validate_config(config)["seed"], 42)
        config["algorithm"]["mutation_rate"] = 2.0
        with self.assertRaises(ConfigurationError):
            validate_config(config)

    def test_manifest_is_serializable_and_persisted(self):
        config = {
            "seed": 42,
            "algorithm": {"name": "test"},
            "search_space": {"choices": [4, 8]},
            "model_name": "fixture",
        }
        manifest = build_manifest(config, {"profile": "fixture.json"})
        self.assertIn("config_hash", manifest)
        self.assertEqual(manifest["config_hash"], configuration_hash(config))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, manifest)
            loaded = json.loads(path.read_text())
        self.assertEqual(loaded["inputs"]["profile"], "fixture.json")

    def test_checkpoint_resume_matches_uninterrupted_steps(self):
        config = {
            "pop_size": 6,
            "n_generations": 2,
            "tournament_size": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "checkpoint_freq": 1,
        }

        with tempfile.TemporaryDirectory() as directory:
            seed_everything(9)
            baseline = GAEngine(
                MixedPrecisionSearchSpace(3, [4, 8]), FixedEvaluator(), config, output_dir=directory
            )
            baseline.population = [baseline.space.sample() for _ in range(config["pop_size"])]
            baseline.step(1)
            baseline.step(2)
            expected_population = [genome.genes for genome in baseline.population]

            seed_everything(9)
            first = GAEngine(
                MixedPrecisionSearchSpace(3, [4, 8]), FixedEvaluator(), config, output_dir=directory
            )
            first.population = [first.space.sample() for _ in range(config["pop_size"])]
            first.step(1)
            checkpoint = Path(directory) / "checkpoint_gen_1.json"
            self.assertTrue(checkpoint.exists())

            resumed = GAEngine(
                MixedPrecisionSearchSpace(3, [4, 8]), FixedEvaluator(), config, output_dir=directory
            )
            next_generation = resumed.load_checkpoint(str(checkpoint))
            self.assertEqual(next_generation, 2)
            resumed.step(next_generation)
            self.assertEqual([genome.genes for genome in resumed.population], expected_population)


if __name__ == "__main__":
    unittest.main()
