"""
Locust Load Test Suite for ArchiGen 3D FastAPI API
Simulates concurrent users hitting the inference & demo endpoints to
validate production readiness (SLA, error rates, throughput).

Run locally:
    locust -f locustfile.py --host http://localhost:8000

Run headless (CI):
    locust -f locustfile.py --host http://localhost:8000 \
      --headless -u 50 -r 10 -t 300s --csv locust_results
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, print_stats
import io
from PIL import Image
import numpy as np


class ArchiGenApiUser(HttpUser):
    """Simulated user exercising the ArchiGen 3D REST API."""
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task(3)
    def health_check(self):
        """GET /health - lightweight availability probe."""
        self.client.get("/health", name="health")

    @task(2)
    def system_info(self):
        """GET /system/info - metadata endpoint."""
        self.client.get("/system/info", name="system_info")

    @task(2)
    def metrics_summary(self):
        """GET /metrics/summary - MLOps metrics."""
        self.client.get("/metrics/summary", name="metrics_summary")

    @task(2)
    def monitoring_drift_status(self):
        """GET /monitoring/drift/status - MLOps Level 3 Drift Probe."""
        self.client.get("/monitoring/drift/status?domain=floorplan", name="drift_status")

    @task(2)
    def monitoring_dvc_status(self):
        """GET /monitoring/dvc/status - DVC Remote Storage Probe."""
        self.client.get("/monitoring/dvc/status", name="dvc_status")

    @task(5)
    def demo_floorplan(self):
        """GET /demo/floorplan - end-to-end segmentation + 3D vectorization."""
        self.client.get("/demo/floorplan", name="demo_floorplan")

    @task(5)
    def demo_facade(self):
        """GET /demo/facade - end-to-end facade synthesis."""
        self.client.get("/demo/facade", name="demo_facade")

    @task(3)
    def predict_floorplan(self):
        """POST /predict/floorplan - upload a synthetic floorplan image."""
        img = Image.fromarray(
            np.random.randint(200, 255, (256, 256, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        self.client.post(
            "/predict/floorplan",
            files={"file": ("test_plan.png", buf, "image/png")},
            name="predict_floorplan",
        )

    @task(3)
    def predict_facade(self):
        """POST /predict/facade - upload a synthetic facade mask."""
        mask = np.zeros((256, 256, 3), dtype=np.uint8)
        mask[50:200, 50:200] = [0, 0, 255]  # facade blue
        mask[100:150, 100:150] = [0, 255, 255]  # window cyan
        img = Image.fromarray(mask)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        self.client.post(
            "/predict/facade",
            files={"file": ("mask.png", buf, "image/png")},
            name="predict_facade",
        )


# ---------------------------------------------------------------------------
# Load test thresholds (fail CI if SLA breached)
# ---------------------------------------------------------------------------
@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs):
    environment.parsed_options = getattr(environment, "parsed_options", None)


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Assert SLA thresholds at end of load test."""
    failures = 0
    for name, stats in environment.runner.stats.entries.items():
        # p95 latency threshold: 2000 ms for CPU inference endpoints
        if stats.get_response_time_percentile(0.95) > 2000:
            print(f"[SLA FAIL] {name}: p95 latency "
                  f"{stats.get_response_time_percentile(0.95):.0f} ms > 2000 ms")
            failures += 1
        # Error rate threshold: < 5%
        if stats.fail_ratio > 0.05:
            print(f"[SLA FAIL] {name}: error ratio {stats.fail_ratio:.2%} > 5%")
            failures += 1
    if failures:
        print("[LOAD TEST] SLA thresholds BREACHED")
        environment.runner.quit()
    else:
        print("[LOAD TEST] All SLA thresholds met")
