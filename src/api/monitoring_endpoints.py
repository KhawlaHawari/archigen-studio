"""
ArchiGen 3D - MLOps Level 3 Drift & DVC Monitoring API Endpoints
TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027
Encadrant : Haythem Ghazouani
Équipe : Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any, Optional

from src.mlops.drift_detector import (
    DriftDetector,
    simulate_production_drift,
    DRIFT_REPORT_PATH,
    REFERENCE_PROFILE_PATH
)
from src.mlops.continuous_retraining import evaluate_and_trigger_ct
from src.mlops.dvc_remote import get_full_dvc_status

router = APIRouter(prefix="/monitoring", tags=["MLOps Level 3 - Drift & Versioning"])


@router.get("/dvc/status")
async def get_dvc_status_endpoint() -> Dict[str, Any]:
    """
    Returns full DVC versioning status:
    configured remotes (MinIO, S3, Local), tracked models and datasets.
    """
    return get_full_dvc_status()


@router.get("/drift/status")
async def get_drift_status_endpoint(domain: str = Query("floorplan", enum=["floorplan", "facade"])) -> Dict[str, Any]:
    """
    Fast health check for data and prediction drift.
    Returns composite drift score, max PSI, and retraining recommendation.
    """
    detector = DriftDetector()
    report = detector.detect_drift_on_buffer(domain=domain)
    return {
        "domain": domain,
        "status": report.get("status", "UNKNOWN"),
        "drift_score": report.get("drift_score", 0.0),
        "drift_detected": report.get("drift_detected", False),
        "retraining_recommended": report.get("retraining_recommended", False),
        "max_psi": report.get("max_psi", 0.0),
        "mean_psi": report.get("mean_psi", 0.0),
        "evaluated_at": report.get("evaluated_at", "N/A"),
        "current_samples_count": report.get("current_samples_count", 0)
    }


@router.get("/drift/report")
async def get_drift_full_report_endpoint(domain: str = Query("floorplan", enum=["floorplan", "facade"])) -> Dict[str, Any]:
    """
    Returns complete statistical drift report:
    PSI per feature, Kolmogorov-Smirnov p-values, Wasserstein distances, and mean shifts.
    """
    detector = DriftDetector()
    report = detector.detect_drift_on_buffer(domain=domain)
    return report


@router.post("/drift/detect")
async def trigger_drift_detection_endpoint(domain: str = Query("floorplan", enum=["floorplan", "facade"])) -> Dict[str, Any]:
    """
    Forces an immediate drift audit over the recent production sliding window.
    Logs drift metrics to MLflow Tracking Server.
    """
    detector = DriftDetector()
    report = detector.detect_drift_on_buffer(domain=domain)
    detector.log_drift_to_mlflow(report)
    return {
        "message": f"Drift audit completed for domain '{domain}'.",
        "result": report
    }


@router.post("/drift/simulate")
async def simulate_drift_endpoint(
    domain: str = Query("floorplan", enum=["floorplan", "facade"]),
    drift_type: str = Query("inverted_dark_mode", enum=["inverted_dark_mode", "heavy_noise", "noise_and_shift"]),
    samples: int = Query(15, ge=5, le=50)
) -> Dict[str, Any]:
    """
    Simulates production drift (e.g. inverted dark-mode plans, noisy scans)
    to test automated drift detection and Continuous Retraining triggers in real-time.
    """
    records = simulate_production_drift(domain=domain, drift_type=drift_type, n_samples=samples)
    detector = DriftDetector()
    report = detector.detect_drift_on_buffer(domain=domain)
    detector.log_drift_to_mlflow(report)

    return {
        "message": f"Successfully injected {len(records)} drifted samples (type: {drift_type}) on domain '{domain}'.",
        "drift_audit_result": report
    }


@router.post("/drift/retrain")
async def trigger_retraining_endpoint(
    domain: str = Query("floorplan", enum=["floorplan", "facade"]),
    force: bool = Query(False, description="Force retraining even if drift threshold is not breached")
) -> Dict[str, Any]:
    """
    MLOps Level 3 Continuous Retraining (CT) Trigger:
    Evaluates drift conditions and launches automated pipeline retraining,
    ONNX validation, and Model Registry recalibration.
    """
    ct_result = evaluate_and_trigger_ct(domain=domain, force=force, dry_run=False)
    return ct_result


@router.get("/drift/dashboard", response_class=HTMLResponse)
async def get_drift_dashboard():
    """
    Serves interactive MLOps Level 3 Visual Monitoring Dashboard.
    """
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArchiGen 3D — Dashboard MLOps Niveau 3 (Drift & Versioning)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: #182234;
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #9b51e0;
            --status-green: #00e676;
            --status-yellow: #ffea00;
            --status-red: #ff1744;
            --text-main: #f0f4f8;
            --text-muted: #94a3b8;
            --border-card: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 24px;
        }
        .header-title h1 {
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-title p {
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-l3 {
            background: linear-gradient(135deg, #7928ca, #ff0080);
            color: #fff;
            box-shadow: 0 0 15px rgba(255, 0, 128, 0.4);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
        }
        .status-pill {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .pill-healthy { background: rgba(0, 230, 118, 0.15); color: var(--status-green); border: 1px solid var(--status-green); }
        .pill-warning { background: rgba(255, 234, 0, 0.15); color: var(--status-yellow); border: 1px solid var(--status-yellow); }
        .pill-critical { background: rgba(255, 23, 68, 0.15); color: var(--status-red); border: 1px solid var(--status-red); }
        .metric-big {
            font-size: 36px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 12px;
            color: var(--text-muted);
        }
        .btn-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            padding: 10px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-blue), #2575fc);
            color: #fff;
        }
        .btn-danger {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: #fff;
        }
        .btn-warning {
            background: linear-gradient(135deg, #f7971e, #ffd200);
            color: #111;
        }
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border-card);
            color: var(--text-main);
        }
        button:hover {
            transform: translateY(-2px);
            opacity: 0.92;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 12px;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        th {
            color: var(--text-muted);
            font-weight: 600;
            background: rgba(255, 255, 255, 0.02);
        }
        td { font-family: 'JetBrains Mono', monospace; }
        .bar-container {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 4px;
        }
        .bar-fill { height: 100%; border-radius: 3px; }
        pre {
            background: #0d131f;
            padding: 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            max-height: 250px;
            overflow-y: auto;
            color: #38ef7d;
        }
        a { color: var(--accent-cyan); text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>🏛️ ArchiGen 3D — Plateforme MLOps Niveau 3</h1>
            <p>TEK-UP UNIVERSITÉ | Deep Learning Avancé & MLOps 2026/2027 · Haythem Ghazouani · Trinôme Ingénieur</p>
        </div>
        <div>
            <span class="badge badge-l3">MLOps Niveau 3 : CT & Drift Monitoring</span>
        </div>
    </div>

    <div class="grid">
        <!-- DVC Remote Status Card -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">📦 DVC Remote Storage (S3 / MinIO)</span>
                <span id="dvc-badge" class="status-pill pill-healthy">INITIALISÉ</span>
            </div>
            <div class="metric-label">Remote Actif par Défaut :</div>
            <div id="dvc-default-remote" class="metric-big" style="font-size: 20px; color: var(--accent-cyan);">minio-storage</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
                URL : <span id="dvc-url" style="color: var(--text-main); font-family: 'JetBrains Mono';">s3://archigen-dvc</span><br>
                Poids Suivis : <span id="dvc-weights" style="color: var(--text-main);">PyTorch (.pt) + ONNX (~250 MB)</span>
            </div>
            <div class="btn-group">
                <button class="btn-outline" onclick="loadDvcStatus()">Rafraîchir DVC</button>
                <a href="/" style="padding: 10px 16px; border: 1px solid var(--border-card); border-radius: 8px; font-weight: 600; font-size: 13px;">Ouvrir Visualiseur 3D</a>
            </div>
        </div>

        <!-- Drift Monitoring Status Card -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">📉 Monitoring de Drift (Population Stability)</span>
                <span id="drift-badge" class="status-pill pill-healthy">HEALTHY</span>
            </div>
            <div class="metric-label">Score Composite de Drift :</div>
            <div id="drift-score" class="metric-big" style="color: var(--status-green);">0.00</div>
            <div style="font-size: 12px; color: var(--text-muted);">
                Max PSI : <span id="max-psi" style="color: var(--text-main); font-family: 'JetBrains Mono';">0.00</span> |
                Échantillons récents : <span id="samples-count" style="color: var(--text-main); font-family: 'JetBrains Mono';">0</span>
            </div>
            <div class="metric-label" style="margin-top: 10px;">Réentraînement Automatique (CT) :</div>
            <div id="retrain-recommendation" style="font-size: 13px; font-weight: 700; color: var(--status-green);">Non requis (Modèle stable)</div>
        </div>

        <!-- Automated CT Controls -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">⚡ Déclencheurs & Simulations MLOps 3</span>
            </div>
            <div class="metric-label">Simuler du trafic de production avec dérive :</div>
            <div class="btn-group">
                <button class="btn-warning" onclick="simulateDrift('inverted_dark_mode')">Simuler Drift (Inversion Nuit)</button>
                <button class="btn-warning" onclick="simulateDrift('heavy_noise')">Simuler Drift (Bruit Scan)</button>
                <button class="btn-primary" onclick="evaluateDrift()">Évaluer Drift</button>
                <button class="btn-danger" onclick="triggerRetraining()">Déclencher CT (Auto-Train)</button>
            </div>
        </div>
    </div>

    <!-- Feature Drift Table -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
            <span class="card-title">🔍 Diagnostic Détaillé des Caractéristiques (PSI & Tests Kolmogorov-Smirnov)</span>
            <span style="font-size: 12px; color: var(--text-muted);">Seuils : PSI &lt; 0.10 (Vert) · 0.10-0.20 (Jaune) · &gt; 0.20 (Rouge / Dérive)</span>
        </div>
        <table id="drift-table">
            <thead>
                <tr>
                    <th>Caractéristique Extraite</th>
                    <th>PSI (Indice Stabilité)</th>
                    <th>Statut PSI</th>
                    <th>KS Stat / p-value</th>
                    <th>Wasserstein Dist.</th>
                    <th>Moyenne Réf. → Actuelle</th>
                </tr>
            </thead>
            <tbody id="drift-table-body">
                <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Chargement des métriques de drift...</td></tr>
            </tbody>
        </table>
    </div>

    <!-- Terminal Output -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">📡 Journal d'Événements MLOps (Live Telemetry & Logs)</span>
        </div>
        <pre id="console-log">Initialisation du système de monitoring MLOps Niveau 3...</pre>
    </div>

    <script>
        function log(msg) {
            const pre = document.getElementById('console-log');
            const time = new Date().toLocaleTimeString();
            pre.textContent = `[${time}] ${msg}\\n` + pre.textContent;
        }

        async function loadDvcStatus() {
            try {
                const res = await fetch('/monitoring/dvc/status');
                const data = await res.json();
                document.getElementById('dvc-default-remote').textContent = data.default_remote || 'Non configuré';
                if (data.remotes && data.remotes[data.default_remote]) {
                    document.getElementById('dvc-url').textContent = data.remotes[data.default_remote].url;
                }
                log('Statut DVC chargé : Remote = ' + data.default_remote);
            } catch (e) {
                log('Erreur chargement DVC : ' + e);
            }
        }

        async function evaluateDrift() {
            log('Évaluation du drift en cours...');
            try {
                const res = await fetch('/monitoring/drift/report?domain=floorplan');
                const data = await res.json();
                renderDriftReport(data);
                log('Audit de drift terminé. Statut = ' + data.status + ', Drift Score = ' + data.drift_score);
            } catch (e) {
                log('Erreur évaluation drift : ' + e);
            }
        }

        async function simulateDrift(type) {
            log('Injection de requêtes avec dérive synthétique (' + type + ')...');
            try {
                const res = await fetch(`/monitoring/drift/simulate?domain=floorplan&drift_type=${type}&samples=15`, { method: 'POST' });
                const json = await res.json();
                log(json.message);
                if (json.drift_audit_result) {
                    renderDriftReport(json.drift_audit_result);
                }
            } catch (e) {
                log('Erreur simulation drift : ' + e);
            }
        }

        async function triggerRetraining() {
            log('Déclenchement du réentraînement continu (CT)...');
            try {
                const res = await fetch('/monitoring/drift/retrain?domain=floorplan&force=true', { method: 'POST' });
                const json = await res.json();
                log('Résultat CT : ' + JSON.stringify(json));
                evaluateDrift();
            } catch (e) {
                log('Erreur déclenchement CT : ' + e);
            }
        }

        function renderDriftReport(data) {
            const badge = document.getElementById('drift-badge');
            const scoreEl = document.getElementById('drift-score');
            const maxPsiEl = document.getElementById('max-psi');
            const samplesEl = document.getElementById('samples-count');
            const retrainEl = document.getElementById('retrain-recommendation');

            scoreEl.textContent = (data.drift_score !== undefined) ? data.drift_score.toFixed(2) : '0.00';
            maxPsiEl.textContent = (data.max_psi !== undefined) ? data.max_psi.toFixed(2) : '0.00';
            samplesEl.textContent = data.current_samples_count || 0;

            if (data.drift_detected) {
                badge.className = 'status-pill pill-critical';
                badge.textContent = 'DRIFT DETECTED';
                scoreEl.style.color = 'var(--status-red)';
                retrainEl.textContent = '🚨 CT RECOMMANDÉ (Réentraînement automatique requis)';
                retrainEl.style.color = 'var(--status-red)';
            } else if (data.status === 'WARNING_DRIFT') {
                badge.className = 'status-pill pill-warning';
                badge.textContent = 'WARNING';
                scoreEl.style.color = 'var(--status-yellow)';
                retrainEl.textContent = '⚠️ Surveillance renforcée (Dérive modérée)';
                retrainEl.style.color = 'var(--status-yellow)';
            } else {
                badge.className = 'status-pill pill-healthy';
                badge.textContent = 'HEALTHY';
                scoreEl.style.color = 'var(--status-green)';
                retrainEl.textContent = '✅ Non requis (Modèle conforme)';
                retrainEl.style.color = 'var(--status-green)';
            }

            // Fill table
            const tbody = document.getElementById('drift-table-body');
            tbody.innerHTML = '';

            const reports = data.feature_reports || {};
            const keys = Object.keys(reports);
            if (keys.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Aucune donnée de feature disponible. Simulez du trafic ou évaluez un plan.</td></tr>';
                return;
            }

            keys.forEach(feat => {
                const f = reports[feat];
                const tr = document.createElement('tr');
                let pillClass = 'pill-healthy';
                if (f.psi_status === 'CRITICAL_DRIFT') pillClass = 'pill-critical';
                else if (f.psi_status === 'MODERATE_DRIFT') pillClass = 'pill-warning';

                const pctWidth = Math.min(100, Math.round(f.psi * 200));
                const barColor = f.psi >= 0.2 ? 'var(--status-red)' : (f.psi >= 0.1 ? 'var(--status-yellow)' : 'var(--status-green)');

                tr.innerHTML = `
                    <td><strong>${feat}</strong></td>
                    <td>
                        ${f.psi.toFixed(3)}
                        <div class="bar-container"><div class="bar-fill" style="width: ${pctWidth}%; background: ${barColor};"></div></div>
                    </td>
                    <td><span class="status-pill ${pillClass}">${f.psi_status}</span></td>
                    <td>${f.ks_statistic.toFixed(3)} (p=${f.ks_p_value.toFixed(4)})</td>
                    <td>${f.wasserstein_distance.toFixed(3)}</td>
                    <td>${f.baseline_mean.toFixed(2)} → ${f.current_mean.toFixed(2)}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Auto load on start
        loadDvcStatus();
        evaluateDrift();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
