from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from pathlib import Path


class NvidiaSMITelemetry:
    """Background nvidia-smi sampler. Fails open if nvidia-smi is unavailable."""

    def __init__(self, run_id: str, out_csv: str | Path, interval_s: float = 0.5):
        self.run_id = run_id
        self.out_csv = Path(out_csv)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.rows: list[dict] = []
        self.start_t = None

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        self.gpu_selector = visible if visible and visible != "-1" else None

    def _query(self):
        cmd = ["nvidia-smi"]
        if self.gpu_selector:
            cmd += ["--id", self.gpu_selector]
        cmd += [
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        text = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        line = text.strip().splitlines()[0]
        vals = [x.strip() for x in next(csv.reader([line]))]
        return vals

    def _loop(self):
        while not self._stop.is_set():
            try:
                vals = self._query()
                self.rows.append(
                    {
                        "run_id": self.run_id,
                        "elapsed_s": time.perf_counter() - self.start_t,
                        "gpu_index": vals[0],
                        "gpu_name": vals[1],
                        "utilization_pct": float(vals[2]),
                        "memory_used_mb": float(vals[3]),
                        "memory_total_mb": float(vals[4]),
                        "power_w": float(vals[5]) if vals[5] not in {"N/A", "[N/A]"} else None,
                        "temperature_c": float(vals[6]),
                    }
                )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.start_t = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.rows:
            import pandas as pd
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(self.rows)
            df.to_csv(
                self.out_csv,
                mode="a",
                header=not self.out_csv.exists(),
                index=False,
            )

    @property
    def mean_utilization(self):
        if not self.rows:
            return None
        return sum(r["utilization_pct"] for r in self.rows) / len(self.rows)

    @property
    def max_memory_mb(self):
        if not self.rows:
            return None
        return max(r["memory_used_mb"] for r in self.rows)
