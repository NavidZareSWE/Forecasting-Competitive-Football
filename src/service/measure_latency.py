"""Bonus (brief section 11): latency distribution and offline parity.

Latency is measured in process through FastAPI's TestClient, so the numbers
describe the service (feature lookup + model evaluation + serialization)
rather than the network stack. Repeated calls, never a single lucky timing.

Parity (the TA's check): the API's calibrated outcome probabilities for a
snapshot must equal the sweep's stored offline predictions for the same
match, minute and model. Both sides fit the same estimator with the same
tuned parameters and seed on the same matrices, so any disagreement means the
service reimplemented something it should have reused - exactly the
deployment bug the brief warns about.

Writes src/reports/api_latency.csv (read by report section 9).

    python src/service/measure_latency.py
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app import app, RESULTS_DIR  # noqa: E402

N_CALLS = {"/health": 300, "/predict": 300, "/predict?pre": 300,
           "/replay": 60}
PARITY_TOLERANCE = 5e-4    # the API rounds probabilities to 4 decimals


def measure(client, label, request):
    calls = N_CALLS[label]
    for _ in range(5):                       # warm-up, excluded
        request()
    samples = []
    for _ in range(calls):
        start = time.perf_counter()
        response = request()
        samples.append((time.perf_counter() - start) * 1000.0)
        assert response.status_code == 200, f"{label} -> {response.status_code}"
    array = np.asarray(samples)
    row = {"endpoint": label, "n": calls,
           "mean_ms": round(float(array.mean()), 3),
           "p50_ms": round(float(np.percentile(array, 50)), 3),
           "p95_ms": round(float(np.percentile(array, 95)), 3),
           "p99_ms": round(float(np.percentile(array, 99)), 3),
           "max_ms": round(float(array.max()), 3)}
    print(f"  {label:16s} p50={row['p50_ms']:8.2f}ms  p95={row['p95_ms']:8.2f}ms  "
          f"p99={row['p99_ms']:8.2f}ms")
    return row


def parity_check(client, match_id, minute):
    """API probabilities vs the sweep's stored offline prediction."""
    from app import state
    model_name = state.models["Lc"].model_name
    stored = pd.read_csv(RESULTS_DIR / "predictions_Lc.csv", encoding="utf-8")
    offline = stored[(stored["model"] == model_name)
                     & (stored["match_id"] == match_id)
                     & (stored["snapshot_minute"] == minute)]
    if offline.empty:
        print(f"  parity: no stored sweep prediction for {model_name} on "
              f"match {match_id} minute {minute}; run run_models.py first")
        return False
    response = client.get(f"/predict?match_id={match_id}&minute={minute}")
    api = response.json()["probabilities"]
    for label, column in [("H", "p_H"), ("D", "p_D"), ("A", "p_A")]:
        gap = abs(api[label] - float(offline[column].iloc[0]))
        assert gap <= PARITY_TOLERANCE, (
            f"parity violation on p_{label}: API {api[label]} vs offline "
            f"{float(offline[column].iloc[0])} (gap {gap:.6f}) - the service "
            "disagrees with the training pipeline")
    print(f"  parity: API == offline sweep prediction for {model_name}, "
          f"match {match_id}, minute {minute} (tolerance {PARITY_TOLERANCE})")
    return True


def main():
    with TestClient(app) as client:          # runs the startup hook
        from app import state
        test_match = int(state.models["C"].frame["match_id"].iloc[0])
        minute = state.minutes[len(state.minutes) // 2]

        checked = parity_check(client, test_match, minute)

        print("Measuring latency (in process)...")
        rows = [
            measure(client, "/health", lambda: client.get("/health")),
            measure(client, "/predict", lambda: client.get(
                f"/predict?match_id={test_match}&minute={minute}")),
            measure(client, "/predict?pre", lambda: client.get(
                f"/predict?match_id={test_match}")),
            measure(client, "/replay", lambda: client.get(
                f"/replay/{test_match}")),
        ]

    latency = pd.DataFrame(rows)
    assert (latency.loc[latency["endpoint"] != "/replay", "p99_ms"]
            < 200).all(), "single-prediction p99 must stay under 200 ms"
    output_path = RESULTS_DIR / "api_latency.csv"
    latency.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {output_path}")
    if not checked:
        sys.exit(1)


if __name__ == "__main__":
    main()
