"""
utils/tracker.py
================
Centroid-based object tracker with ID assignment and disappearance handling.
"""

import numpy as np
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_id = 0
        self.objects = OrderedDict()      # id -> (bbox, cls, conf)
        self.centroids = OrderedDict()    # id -> (cx, cy)
        self.disappeared = OrderedDict()  # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, cls, conf):
        self.objects[self.next_id] = (bbox, cls, conf)
        self.centroids[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.centroids[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections: list) -> OrderedDict:
        """
        detections: list of ([x1,y1,x2,y2], conf, cls_name)
        returns: OrderedDict of id -> (bbox, cls, conf)
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Compute input centroids
        input_centroids = []
        for (bbox, conf, cls) in detections:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            input_centroids.append((cx, cy))

        if len(self.centroids) == 0:
            for i, (bbox, conf, cls) in enumerate(detections):
                self.register(input_centroids[i], bbox, cls, conf)
        else:
            obj_ids = list(self.centroids.keys())
            obj_cents = list(self.centroids.values())

            # Compute distance matrix
            D = np.zeros((len(obj_cents), len(input_centroids)))
            for i, oc in enumerate(obj_cents):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((oc[0] - ic[0])**2 + (oc[1] - ic[1])**2)

            # Greedy match by minimum distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                obj_id = obj_ids[row]
                bbox, conf, cls = detections[col]
                self.objects[obj_id] = (bbox, cls, conf)
                self.centroids[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(obj_cents))) - used_rows
            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            # Register new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                bbox, conf, cls = detections[col]
                self.register(input_centroids[col], bbox, cls, conf)

        return self.objects