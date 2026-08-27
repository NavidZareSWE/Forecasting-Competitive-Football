"""Bonus (brief section 11): drive a held-out match through the service in
match order, as if it were happening now.

Feeds the service one snapshot request per tick and prints the evolving
forecast; open http://127.0.0.1:5500/ for the visual dashboard version.

The tick schedule is a wall-clock deadline, not sleep-after-request, so the
printed cadence stays at `pace` instead of drifting by the request latency.
Each line reports that latency so the demo shows what the service costs.

    python src/service/app.py                          # terminal 1
    python src/service/replay_driver.py                # terminal 2 (first match)
    python src/service/replay_driver.py 3825848 0.5    # match id, seconds/tick
"""

import os
import sys
import time

import httpx

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '5500')}"


def main():
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    pace = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    with httpx.Client(base_url=BASE, timeout=30.0) as client:
        if match_id is None:
            match_id = client.get("/matches").json()[0]["match_id"]

        pre = client.get(f"/predict?match_id={match_id}").json()
        p = pre["probabilities"]
        print(f"match {match_id} | pre-match       "
              f"H {p['H']:.3f}  D {p['D']:.3f}  A {p['A']:.3f} | "
              f"margin {pre['expected_margin']:+.2f}")

        minutes = client.get("/minutes").json()["minutes"]
        deadline = time.perf_counter()
        for minute in minutes:
            start = time.perf_counter()
            snap = client.get(
                f"/predict?match_id={match_id}&minute={minute}").json()
            elapsed = (time.perf_counter() - start) * 1000.0
            p = snap["probabilities"]
            top = snap["top_shap"][0]
            print(f"match {match_id} | {minute:3d}'  "
                  f"{snap['score']['home']}-{snap['score']['away']}  "
                  f"H {p['H']:.3f}  D {p['D']:.3f}  A {p['A']:.3f} | "
                  f"margin {snap['expected_margin']:+.2f} | "
                  f"top shap: {top['feature']} ({top['shap']:+.3f}) | "
                  f"{elapsed:.1f} ms")
            deadline += pace
            time.sleep(max(0.0, deadline - time.perf_counter()))


if __name__ == "__main__":
    main()
