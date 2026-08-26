"""Tests for Phase 5 and Phase 6: benchmarking and proxy calibration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.benchmark.cache import MeasurementCache, hardware_fingerprint
from src.benchmark.harness import (
    BenchmarkError,
    BenchmarkResult,
    LatencyStats,
    MemoryStats,
    QualityStats,
    ThroughputStats,
    get_hardware_identity,
    percentile,
)
from src.benchmark.proxy_calibration import ProxyCalibration
from src.benchmark.proxy_validator import ProxyValidator, measure_initial_design_of_experiments
from src.core.evaluator import FitnessMetrics


class Phase5HarnessTests(unittest.TestCase):
    def test_hardware_identity_requires_cuda(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaises(BenchmarkError),
        ):
            get_hardware_identity()

    def test_latency_stats_computes_percentiles(self):
        stats = LatencyStats(
            mean_ms=50,
            std_ms=29.15,
            min_ms=0,
            max_ms=99,
            p50_ms=50,
            p95_ms=95,
            p99_ms=99,
        )
        self.assertEqual(stats.p50_ms, 50)
        self.assertEqual(stats.p95_ms, 95)
        self.assertEqual(stats.p99_ms, 99)

    def test_memory_stats_tracks_incremental(self):
        memory = MemoryStats(baseline_mb=1000, peak_allocated_mb=2000, peak_reserved_mb=2500, incremental_mb=1000)
        self.assertEqual(memory.incremental_mb, 1000)

    def test_throughput_stats_computes_rates(self):
        throughput = ThroughputStats(tokens_per_sec=1000, batches_per_sec=10)
        self.assertEqual(throughput.tokens_per_sec, 1000)
        self.assertEqual(throughput.batches_per_sec, 10)

    def test_quality_stats_holds_loss_and_perplexity(self):
        quality = QualityStats(mean_loss=2.5, perplexity=np.exp(2.5), loss_samples=128)
        self.assertAlmostEqual(quality.perplexity, np.exp(2.5))
        self.assertEqual(quality.loss_samples, 128)

    def test_benchmark_result_serializes(self):
        result = BenchmarkResult(
            genome=[4, 8, 16],
            backend="bitsandbytes",
            model_name="test_model",
            batch_size=1,
            sequence_length=128,
            warmup_iterations=2,
            benchmark_iterations=5,
            load_memory=MemoryStats(baseline_mb=100, peak_allocated_mb=1000, peak_reserved_mb=1200, incremental_mb=900),
            inference_memory=MemoryStats(baseline_mb=900, peak_allocated_mb=1100, peak_reserved_mb=1300, incremental_mb=200),
            inference_latency=LatencyStats(mean_ms=50, std_ms=2, min_ms=48, max_ms=55, p50_ms=50),
            inference_throughput=ThroughputStats(tokens_per_sec=2560, batches_per_sec=20),
            validation_quality=QualityStats(mean_loss=2.5, perplexity=np.exp(2.5), loss_samples=128),
        )
        serialized = result.to_dict()
        self.assertEqual(serialized["genome"], [4, 8, 16])
        self.assertEqual(serialized["backend"], "bitsandbytes")

    def test_percentile_computes_correct_values(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # For 10 items, p50 is at index 5 (0-based), which is value 6
        self.assertEqual(percentile(data, 50), 6)
        self.assertEqual(percentile(data, 0), 1)
        self.assertEqual(percentile(data, 100), 10)
        self.assertIsNone(percentile([], 50))
        self.assertIsNone(percentile(data, -1))
        # Test with 100 items for precise percentiles
        data_100 = list(range(1, 101))
        self.assertGreater(percentile(data_100, 95), 90)


class Phase5CacheTests(unittest.TestCase):
    def test_hardware_fingerprint_is_consistent(self):
        hardware = {
            "gpu_name": "RTX 4070",
            "compute_capability": (8, 9),
            "total_memory_bytes": 8588886016,
            "cuda_version": "12.8",
        }
        fp1 = hardware_fingerprint(hardware)
        fp2 = hardware_fingerprint(hardware)
        self.assertEqual(fp1, fp2)

    def test_hardware_fingerprint_changes_with_hardware(self):
        hw1 = {
            "gpu_name": "RTX 4070",
            "compute_capability": (8, 9),
            "total_memory_bytes": 8588886016,
            "cuda_version": "12.8",
        }
        hw2 = {
            "gpu_name": "RTX 3090",
            "compute_capability": (8, 6),
            "total_memory_bytes": 24576000000,
            "cuda_version": "12.8",
        }
        self.assertNotEqual(hardware_fingerprint(hw1), hardware_fingerprint(hw2))

    def test_measurement_cache_stores_and_retrieves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MeasurementCache(tmpdir)
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }
            key = cache.get_key([4, 8, 16], "bitsandbytes", hardware, batch_size=1, sequence_length=128)
            result = {"genome": [4, 8, 16], "latency_ms": 50}
            cache.store(key, result)
            retrieved = cache.retrieve(key)
            self.assertEqual(retrieved["genome"], [4, 8, 16])

    def test_measurement_cache_key_is_unique(self):
        hardware = {
            "gpu_name": "test",
            "compute_capability": (8, 9),
            "total_memory_bytes": 1000,
            "cuda_version": "12.8",
        }
        cache = MeasurementCache(tempfile.mkdtemp())
        key1 = cache.get_key([4, 8], "bitsandbytes", hardware, batch_size=1, sequence_length=128)
        key2 = cache.get_key([4, 8], "bitsandbytes", hardware, batch_size=2, sequence_length=128)
        self.assertNotEqual(key1.as_filename(), key2.as_filename())


class Phase6ProxyCalibrationTests(unittest.TestCase):
    def test_proxy_calibration_adds_points(self):
        calibration = ProxyCalibration()
        hardware = {
            "gpu_name": "test",
            "compute_capability": (8, 9),
            "total_memory_bytes": 1000,
            "cuda_version": "12.8",
        }
        calibration.add_point(
            genome=[4, 8, 16],
            predicted_quality_loss=2.5,
            measured_loss=2.3,
            measured_perplexity=10.0,
            measured_memory_mb=1000,
            measured_latency_ms=50,
            measured_throughput=2560,
            hardware_identity=hardware,
        )
        self.assertEqual(len(calibration.calibration_points), 1)
        point = calibration.calibration_points[0]
        self.assertEqual(point.genome, [4, 8, 16])
        self.assertAlmostEqual(point.proxy.predicted_quality_loss, 2.5)
        self.assertAlmostEqual(point.measured.mean_loss, 2.3)

    def test_proxy_calibration_computes_correlation(self):
        calibration = ProxyCalibration()
        hardware = {"gpu_name": "test", "compute_capability": (8, 9), "total_memory_bytes": 1000, "cuda_version": "12.8"}
        
        for i in range(10):
            calibration.add_point(
                genome=[i, 8, 16],
                predicted_quality_loss=2.0 + i * 0.1,
                measured_loss=2.1 + i * 0.1,
                measured_perplexity=8.0 + i * 0.5,
                measured_memory_mb=1000 - i * 10,
                measured_latency_ms=50 + i * 2,
                measured_throughput=2560 - i * 100,
                hardware_identity=hardware,
            )
        
        report = calibration.compute_correlation()
        self.assertEqual(report["calibration_point_count"], 10)
        self.assertIn("proxy_to_measured_loss_correlation", report)
        self.assertIsNotNone(report["proxy_to_measured_loss_correlation"])

    def test_proxy_calibration_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = ProxyCalibration("v1")
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }
            calibration.add_point(
                genome=[4, 8],
                predicted_quality_loss=2.5,
                measured_loss=2.3,
                measured_perplexity=10.0,
                measured_memory_mb=1000,
                measured_latency_ms=50,
                measured_throughput=2560,
                hardware_identity=hardware,
            )
            calibration.compute_correlation()
            report_path = Path(tmpdir) / "calibration.json"
            calibration.save_report(report_path)
            
            loaded = ProxyCalibration()
            loaded.load_report(report_path)
            self.assertEqual(len(loaded.calibration_points), 1)
            self.assertEqual(loaded.calibration_points[0].genome, [4, 8])

    def test_insufficient_points_returns_insufficient_status(self):
        calibration = ProxyCalibration()
        report = calibration.compute_correlation()
        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["calibration_point_count"], 0)


class Phase6ProxyValidatorTests(unittest.TestCase):
    def test_proxy_validator_predicts_quality_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ProxyValidator(tmpdir)
            # Uniform FP16 should have minimal quality loss
            loss_fp16 = validator.predict_quality_loss([16] * 22, "tinyllama", 2.0)
            # Uniform INT4 should have higher quality loss
            loss_int4 = validator.predict_quality_loss([4] * 22, "tinyllama", 2.0)
            self.assertLess(loss_fp16, loss_int4)

    def test_proxy_validator_measures_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ProxyValidator(tmpdir)
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }
            fitness = FitnessMetrics(mean_loss=2.3, peak_memory_mb=1000, latency_ms=50)
            validator.measure_candidate([4, 8], fitness, 2.5, hardware)
            self.assertEqual(len(validator.calibration.calibration_points), 1)

    def test_proxy_validator_saves_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "calibration.json"
            validator = ProxyValidator(tmpdir, calibration_report_path=report_path)
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }
            fitness = FitnessMetrics(mean_loss=2.3, peak_memory_mb=1000, latency_ms=50)
            validator.measure_candidate([4, 8], fitness, 2.5, hardware)
            validator.save_calibration()
            self.assertTrue(report_path.exists())

    def test_measure_initial_design_of_experiments(self):
        mock_fitness = FitnessMetrics(mean_loss=2.5, peak_memory_mb=1000, latency_ms=50)
        mock_validator = MagicMock(return_value=mock_fitness)
        results = measure_initial_design_of_experiments(mock_validator, num_layers=22, hardware_identity={})
        self.assertIn("uniform_fp16", results)
        self.assertIn("uniform_int8", results)
        self.assertIn("uniform_int4", results)
        self.assertEqual(len(results["uniform_fp16"]["genome"]), 22)


class Phase5And6IntegrationTests(unittest.TestCase):
    def test_cache_proxy_workflow(self):
        """Test end-to-end workflow: benchmark → cache → proxy validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MeasurementCache(tmpdir)
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }

            # Step 1: Store a benchmark result
            key = cache.get_key([4, 8], "bitsandbytes", hardware, 1, 128)
            result = {"genome": [4, 8], "latency_ms": 50, "memory_mb": 1000}
            cache.store(key, result)

            # Step 2: Retrieve and verify
            retrieved = cache.retrieve(key)
            self.assertEqual(retrieved["latency_ms"], 50)

            # Step 3: Validate proxy against this measurement
            proxy_validator = ProxyValidator(tmpdir)
            fitness = FitnessMetrics(mean_loss=2.3, peak_memory_mb=1000, latency_ms=50)
            proxy_validator.measure_candidate([4, 8], fitness, 2.5, hardware)
            correlation = proxy_validator.validate_proxy()
            self.assertEqual(correlation["calibration_point_count"], 1)

    def test_predicted_vs_measured_schema_separation(self):
        """Verify that predicted and measured metrics use different fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration = ProxyCalibration()
            hardware = {
                "gpu_name": "test",
                "compute_capability": (8, 9),
                "total_memory_bytes": 1000,
                "cuda_version": "12.8",
            }
            calibration.add_point(
                genome=[4, 8],
                predicted_quality_loss=2.5,
                measured_loss=2.3,
                measured_perplexity=10.0,
                measured_memory_mb=1000,
                measured_latency_ms=50,
                measured_throughput=2560,
                hardware_identity=hardware,
            )
            report_path = Path(tmpdir) / "report.json"
            calibration.save_report(report_path)
            data = json.loads(report_path.read_text())
            point = data["calibration_points"][0]
            # Verify schema separation
            self.assertIn("predicted_quality_loss", point["proxy"])
            self.assertIn("mean_loss", point["measured"])
            self.assertNotIn("mean_loss", point["proxy"])
            self.assertNotIn("predicted_quality_loss", point["measured"])

    def test_cache_key_changes_with_backend(self):
        """Verify cache keys change when backend changes."""
        hardware = {
            "gpu_name": "test",
            "compute_capability": (8, 9),
            "total_memory_bytes": 1000,
            "cuda_version": "12.8",
        }
        cache = MeasurementCache(tempfile.mkdtemp())
        key1 = cache.get_key([4, 8], "bitsandbytes", hardware, 1, 128)
        key2 = cache.get_key([4, 8], "simulation", hardware, 1, 128)
        self.assertNotEqual(key1.as_filename(), key2.as_filename())

    def test_cache_key_changes_with_hardware(self):
        """Verify cache keys change when hardware changes."""
        hw1 = {
            "gpu_name": "RTX 4070",
            "compute_capability": (8, 9),
            "total_memory_bytes": 8000000000,
            "cuda_version": "12.8",
        }
        hw2 = {
            "gpu_name": "RTX 3090",
            "compute_capability": (8, 6),
            "total_memory_bytes": 24000000000,
            "cuda_version": "12.8",
        }
        cache = MeasurementCache(tempfile.mkdtemp())
        key1 = cache.get_key([4, 8], "bitsandbytes", hw1, 1, 128)
        key2 = cache.get_key([4, 8], "bitsandbytes", hw2, 1, 128)
        self.assertNotEqual(key1.as_filename(), key2.as_filename())


if __name__ == "__main__":
    unittest.main()
