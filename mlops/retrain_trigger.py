"""Publish retraining event (mocks AWS EventBridge if no creds)."""
from __future__ import annotations

import argparse
import json
import logging
import os

log = logging.getLogger("retrain_trigger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def trigger(event: dict, bus: str = "ml-churn-bus") -> dict:
    payload = {
        "Source": "churn_model.drift",
        "DetailType": "RetrainingTriggered",
        "Detail": json.dumps(event),
        "EventBusName": bus,
    }
    if os.getenv("AWS_DEFAULT_REGION"):
        try:
            import boto3
            return boto3.client("events").put_events(Entries=[payload])
        except Exception as exc:
            log.warning("EventBridge publish failed (%s); logging only", exc)
    log.info("[mock] would publish: %s", json.dumps(payload, indent=2))
    return {"mock": True, "payload": payload}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-report", required=True)
    p.add_argument("--bus", default="ml-churn-bus")
    args = p.parse_args()
    rep = json.loads(open(args.drift_report).read())
    if not rep.get("trigger_retrain"):
        log.info("No drift trigger; skipping.")
        return
    trigger({"reason": "feature_drift", "drift_report": rep}, bus=args.bus)


if __name__ == "__main__":
    main()
