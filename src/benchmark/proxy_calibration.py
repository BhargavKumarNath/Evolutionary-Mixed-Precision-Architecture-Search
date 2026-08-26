"""Proxy model calibration and validation against held-out data."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyMetrics:
    """Predicted quality loss from the proxy model."""

    predicted_quality_loss: float
    model_version: str


@dataclass(frozen=True)
class MeasuredMetrics:
    """Actual hardware measurements."""

    mean_loss: float
    perplexity: float
    peak_memory_mb: float
    latency_ms: float
    throughput_tokens_per_sec: float


@dataclass(frozen=True)
class CalibrationPoint:
    """A candidate with both proxy and measured metrics."""

    genome: list[int]
    proxy: ProxyMetrics
    measured: MeasuredMetrics
    hardware_identity: dict[str, Any]


class ProxyCalibration:
    """Build and validate a proxy model against measured candidates."""

    def __init__(self, model_version: str = "v1"):
        self.model_version = model_version
        self.calibration_points: list[CalibrationPoint] = []
        self.correlation_report: dict[str, Any] = {}

    def add_point(
        self,
        genome: list[int],
        predicted_quality_loss: float,
        measured_loss: float,
        measured_perplexity: float,
        measured_memory_mb: float,
        measured_latency_ms: float,
        measured_throughput: float,
        hardware_identity: dict[str, Any],
    ) -> None:
        """Record a calibration measurement."""
        point = CalibrationPoint(
            genome=genome,
            proxy=ProxyMetrics(predicted_quality_loss=predicted_quality_loss, model_version=self.model_version),
            measured=MeasuredMetrics(
                mean_loss=measured_loss,
                perplexity=measured_perplexity,
                peak_memory_mb=measured_memory_mb,
                latency_ms=measured_latency_ms,
                throughput_tokens_per_sec=measured_throughput,
            ),
            hardware_identity=hardware_identity,
        )
        self.calibration_points.append(point)
        logger.info("Added calibration point: genome=%s predicted=%.4f measured=%.4f", genome, predicted_quality_loss, measured_loss)

    def compute_correlation(self) -> dict[str, Any]:
        """Compute correlations between proxy and measured metrics."""
        base_report = {
            "model_version": self.model_version,
            "calibration_point_count": len(self.calibration_points),
        }

        if len(self.calibration_points) < 2:
            logger.warning("Insufficient calibration points for correlation analysis")
            return {**base_report, "status": "insufficient_data"}

        proxy_values = [p.proxy.predicted_quality_loss for p in self.calibration_points]
        measured_loss = [p.measured.mean_loss for p in self.calibration_points]
        measured_memory = [p.measured.peak_memory_mb for p in self.calibration_points]
        measured_latency = [p.measured.latency_ms for p in self.calibration_points]

        def pearson_correlation(x: list[float], y: list[float]) -> float | None:
            if len(x) < 2 or len(y) < 2:
                return None
            x_arr = np.array(x, dtype=np.float64)
            y_arr = np.array(y, dtype=np.float64)
            x_std, y_std = x_arr.std(), y_arr.std()
            if x_std == 0 or y_std == 0:
                return None
            return float(np.corrcoef(x_arr, y_arr)[0, 1])

        self.correlation_report = {
            **base_report,
            "proxy_to_measured_loss_correlation": pearson_correlation(proxy_values, measured_loss),
            "proxy_to_measured_memory_correlation": pearson_correlation(proxy_values, measured_memory),
            "proxy_to_measured_latency_correlation": pearson_correlation(proxy_values, measured_latency),
            "proxy_mean": float(np.mean(proxy_values)) if proxy_values else None,
            "proxy_std": float(np.std(proxy_values)) if proxy_values else None,
            "measured_loss_mean": float(np.mean(measured_loss)) if measured_loss else None,
            "measured_loss_std": float(np.std(measured_loss)) if measured_loss else None,
            "measured_memory_mean": float(np.mean(measured_memory)) if measured_memory else None,
            "measured_memory_std": float(np.std(measured_memory)) if measured_memory else None,
            "measured_latency_mean": float(np.mean(measured_latency)) if measured_latency else None,
            "measured_latency_std": float(np.std(measured_latency)) if measured_latency else None,
        }
        return self.correlation_report

    def save_report(self, output_path: str | Path) -> None:
        """Save calibration points and correlation report."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model_version": self.model_version,
            "calibration_points": [
                {
                    "genome": p.genome,
                    "proxy": asdict(p.proxy),
                    "measured": asdict(p.measured),
                }
                for p in self.calibration_points
            ],
            "correlation": self.correlation_report,
        }

        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved proxy calibration report: %s", target)

    def load_report(self, input_path: str | Path) -> None:
        """Load calibration points from a saved report."""
        target = Path(input_path)
        if not target.exists():
            logger.info("Calibration report does not exist yet: %s", target)
            return
        data = json.loads(target.read_text(encoding="utf-8"))

        self.model_version = data.get("model_version", "v1")
        for point_data in data.get("calibration_points", []):
            self.add_point(
                genome=point_data["genome"],
                predicted_quality_loss=point_data["proxy"]["predicted_quality_loss"],
                measured_loss=point_data["measured"]["mean_loss"],
                measured_perplexity=point_data["measured"]["perplexity"],
                measured_memory_mb=point_data["measured"]["peak_memory_mb"],
                measured_latency_ms=point_data["measured"]["latency_ms"],
                measured_throughput=point_data["measured"]["throughput_tokens_per_sec"],
                hardware_identity=point_data.get("hardware_identity", {}),
            )
        self.correlation_report = data.get("correlation", {})
