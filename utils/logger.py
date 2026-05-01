"""
utils/logger.py
===============
CSV and JSON event logger for all perimeter crossing events.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class EventLogger:
    def __init__(self, output_dir: str = "outputs/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"events_{timestamp}.csv"
        self.json_path = self.output_dir / f"events_{timestamp}.json"

        self.events = []
        self.stats = defaultdict(int)

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "frame", "track_id", "class", "line", "direction"
            ])
            writer.writeheader()

    def log(self, event: dict):
        self.events.append(event)
        self.stats[event["class"]] += 1

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "frame", "track_id", "class", "line", "direction"
            ])
            writer.writerow({
                "timestamp": event["timestamp"],
                "frame":     event["frame"],
                "track_id":  event["track_id"],
                "class":     event["class"],
                "line":      event["line"],
                "direction": event["direction"]
            })

        # Rewrite JSON (small overhead, keeps it always valid)
        with open(self.json_path, "w") as f:
            json.dump(self.events, f, indent=2)

    def summary(self):
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Total crossing events : {len(self.events)}")
        for cls, count in self.stats.items():
            print(f"  {cls:<15}: {count}")
        print(f"\nCSV log  : {self.csv_path}")
        print(f"JSON log : {self.json_path}")
        print("="*50)