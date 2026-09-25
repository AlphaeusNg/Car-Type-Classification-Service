"""Process-local aggregates for prediction work.

Snapshots contain counts and durations only. They must not include image
bytes, filesystem paths, or request payloads.
"""

from __future__ import annotations

from threading import Lock


class TimingAggregate:
    """Count, total, and maximum of one elapsed-time series."""

    def __init__(self) -> None:
        self.count = 0
        self.total_seconds = 0.0
        self.max_seconds = 0.0

    def record(self, seconds: float) -> None:
        duration = seconds if seconds > 0 else 0.0
        self.count += 1
        self.total_seconds += duration
        if duration > self.max_seconds:
            self.max_seconds = duration

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "max_seconds": self.max_seconds,
        }


class RequestMetrics:
    """Aggregate preprocessing, queue wait, inference, rejections, and 503s."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.timings = {
            "preprocessing": TimingAggregate(),
            "inference_queue_wait": TimingAggregate(),
            "inference": TimingAggregate(),
        }
        self.rejections = {
            "unsupported_media_type": 0,
            "empty_image": 0,
            "oversized_upload": 0,
            "oversized_request": 0,
            "invalid_image": 0,
        }
        self.unavailable = {
            "model_not_ready": 0,
            "image_processing_busy": 0,
            "prediction_queue_busy": 0,
        }

    def reset(self) -> None:
        with self._lock:
            self.timings = {name: TimingAggregate() for name in self.timings}
            self.rejections = {name: 0 for name in self.rejections}
            self.unavailable = {name: 0 for name in self.unavailable}

    def record_timing(self, name: str, seconds: float) -> None:
        with self._lock:
            self.timings[name].record(seconds)

    def record_rejection(self, name: str) -> None:
        with self._lock:
            self.rejections[name] += 1

    def record_unavailable(self, name: str) -> None:
        with self._lock:
            self.unavailable[name] += 1

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                "timings": {
                    name: aggregate.snapshot() for name, aggregate in self.timings.items()
                },
                "rejections": dict(self.rejections),
                "unavailable": dict(self.unavailable),
            }
