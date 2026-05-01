"""
utils/alert.py
==============
Alert system: sound beep, save snapshot, optional webhook/email notification.
"""

import cv2
import os
import json
import threading
from datetime import datetime
from pathlib import Path


class AlertSystem:
    def __init__(self, config: dict):
        self.config = config
        self.snapshot_dir = Path(config.get("snapshot_dir", "outputs/snapshots"))
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = config.get("cooldown_seconds", 3)
        self.last_alert = {}  # track_id -> last alert timestamp



    def trigger(self, event: dict, frame):
        print(f"[DEBUG] AlertSystem.trigger() called for track_id={event['track_id']} class={event['class']}")
        track_id = event["track_id"]
        now = datetime.now().timestamp()

        # Cooldown check per track
        if track_id in self.last_alert:
            if now - self.last_alert[track_id] < self.cooldown:
                return
        self.last_alert[track_id] = now

        # Save snapshot
        if self.config.get("save_snapshots", True):
            self._save_snapshot(frame, event)

        # Terminal bell (beep)
        if self.config.get("beep", True):
            threading.Thread(target=self._beep, daemon=True).start()

        # Webhook (optional)
        if self.config.get("webhook_url"):
            threading.Thread(
                target=self._send_webhook,
                args=(event,),
                daemon=True
            ).start()

    def _save_snapshot(self, frame, event):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alert_{event['class']}_id{event['track_id']}_{ts}.jpg"
        path = self.snapshot_dir / filename
        cv2.imwrite(str(path), frame)

    def _beep(self):
        try:
            import winsound
            winsound.Beep(1000, 600)  # 1000Hz frequency, 600ms duration
        except Exception:
            print("\a", end="", flush=True)

    def _send_webhook(self, event: dict):
        try:
            import urllib.request
            data = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                self.config["webhook_url"],
                data=data,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            print(f"[WARN] Webhook failed: {e}")