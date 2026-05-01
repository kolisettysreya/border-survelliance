"""
utils/line_crossing.py
======================
Virtual perimeter line with crossing detection using sign-change method.
"""

import cv2
import numpy as np
from collections import defaultdict


class PerimeterLine:
    def __init__(self, name: str, start: tuple, end: tuple,
                 direction: str = "both", color: tuple = (0, 0, 255)):
        """
        name      : Label for this perimeter line
        start/end : (x, y) pixel coordinates
        direction : 'both' | 'AtoB' | 'BtoA'
        color     : BGR color for visualization
        """
        self.name = name
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.direction = direction
        self.color = color
        self.prev_positions = defaultdict(lambda: None)
        self.cross_count = defaultdict(int)

    def _side(self, point: np.ndarray) -> float:
        """
        Returns positive if point is on left side of line (start→end),
        negative if right side, 0 if on the line.
        """
        d = self.end - self.start
        return (d[0]) * (point[1] - self.start[1]) - (d[1]) * (point[0] - self.start[0])

    def check_crossing(self, track_id: int, centroid: tuple):
        """
        Returns (crossed: bool, direction: str)
        direction: 'AtoB' or 'BtoA'
        """
        point = np.array(centroid, dtype=float)
        prev = self.prev_positions[track_id]
        self.prev_positions[track_id] = point

        if prev is None:
            return False, None

        prev_side = self._side(prev)
        curr_side = self._side(point)

        # Crossing detected when sign flips
        if prev_side * curr_side < 0:
            cross_dir = "AtoB" if prev_side > 0 else "BtoA"

            if self.direction == "both" or self.direction == cross_dir:
                self.cross_count[track_id] += 1
                return True, cross_dir

        return False, None

    def draw(self, frame):
        """Draw perimeter line with label and crossing arrow on frame."""
        h, w = frame.shape[:2]
        p1 = tuple(map(int, self.start))
        p2 = tuple(map(int, self.end))

        # Dashed line effect
        total_len = np.linalg.norm(self.end - self.start)
        dash_len = 20
        gap_len = 10
        num_segments = int(total_len / (dash_len + gap_len))
        direction_vec = (self.end - self.start) / (total_len + 1e-9)

        for i in range(num_segments):
            seg_start = self.start + direction_vec * i * (dash_len + gap_len)
            seg_end = seg_start + direction_vec * dash_len
            cv2.line(frame,
                     tuple(map(int, seg_start)),
                     tuple(map(int, seg_end)),
                     self.color, 2)

        # Line label
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        (tw, th), _ = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,
                      (mid[0] - 4, mid[1] - th - 6),
                      (mid[0] + tw + 4, mid[1] + 2),
                      self.color, -1)
        cv2.putText(frame, self.name, (mid[0], mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Direction arrow at midpoint
        perp = np.array([-(self.end[1] - self.start[1]),
                          (self.end[0] - self.start[0])], dtype=float)
        perp = perp / (np.linalg.norm(perp) + 1e-9) * 20
        arrow_end = (int(mid[0] + perp[0]), int(mid[1] + perp[1]))
        cv2.arrowedLine(frame, mid, arrow_end, self.color, 2, tipLength=0.4)