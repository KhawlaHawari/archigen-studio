# Kubernetes deployment (orchestration + auto-scaling)

| File | What it does |
| :--- | :--- |
| `namespace.yaml` | `archigen` namespace |
| `configmap.yaml` | Runtime settings: A/B champion / challenger / canary share, MLflow URI, threads |
| `models-pvc.yaml` | Volume with the model weights (not in Git, not in the image) |
| `deployment.yaml` | API pods (image from CI: `ghcr.io/khawlahawari/archigen-studio`), rolling updates with zero downtime, startup/readiness/liveness probes on `/health`, Prometheus scrape annotations |
| `service.yaml` | Stable in-cluster address `archigen-api:8000` |
| `hpa.yaml` | **Auto-scaling 2 → 6 pods** at 70 % CPU / 85 % memory, slow scale-down (no flapping) |
| `pdb.yaml` | Keeps at least one pod during node maintenance |

## Try it on your PC (Docker Desktop)
1. Docker Desktop → Settings → **Kubernetes → Enable Kubernetes** (one-node cluster, takes a few minutes).
2. Deploy and copy the weights into the volume:
   ```bash
   kubectl apply -k k8s/
   kubectl -n archigen get pods -w                 # wait for a pod, then copy the models:
   kubectl -n archigen cp models/checkpoints <pod-name>:/app/models/checkpoints
   kubectl -n archigen cp models/onnx        <pod-name>:/app/models/onnx
   kubectl -n archigen rollout restart deployment/archigen-api
   ```
   (The models volume is mounted read-only in the API; for the copy, temporarily remove `readOnly: true`
   or use a helper pod. In production the volume is filled from the GitHub Release / DVC remote.)
3. Reach the API: `kubectl -n archigen port-forward svc/archigen-api 8000:8000`
4. Watch auto-scaling under load (`locust -f locustfile.py`): `kubectl -n archigen get hpa -w`
   (metrics-server is required: `kubectl top pods` must work).

CI (`infra-validate` job in `.github/workflows/mlops-ci.yml`) validates these manifests against the
Kubernetes 1.30 schemas with kubeconform on every push.
