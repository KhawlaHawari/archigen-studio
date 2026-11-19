"""Recover wall axes from pixel evidence without extruding room contour holes."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WallRun:
    start: np.ndarray
    end: np.ndarray
    thickness: float

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self):
        return (self.end - self.start) / self.length

    def outline(self):
        dx, dy = self.direction
        side = np.array([-dy, dx]) * self.thickness / 2
        return np.array([self.start + side, self.end + side, self.end - side, self.start - side])


def structural_support(walls, openings):
    """Include opening pixels only when they connect to a wall in the model mask."""
    near_wall = cv2.dilate(walls, np.ones((5, 5), dtype=np.uint8))
    _, labels = cv2.connectedComponents(openings)
    supported = np.unique(labels[(near_wall > 0) & (openings > 0)])
    structure = walls | np.isin(labels, supported).astype(np.uint8)
    return cv2.morphologyEx(structure, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))


def _thin(mask):
    """Thin thick wall strokes to one-pixel axes while preserving their junctions."""
    pixels = np.pad(mask, 1)
    core = pixels[1:-1, 1:-1]
    neighbors = (
        pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:], pixels[2:, 2:],
        pixels[2:, 1:-1], pixels[2:, :-2], pixels[1:-1, :-2], pixels[:-2, :-2],
    )
    north, _, east, _, south, _, west, _ = neighbors
    while True:
        removed = False
        for corner_a, corner_b in (((north, east, south), (east, south, west)),
                                   ((north, east, west), (north, south, west))):
            count = sum(neighbors)
            transitions = sum((b > a).astype(np.uint8) for a, b in zip(neighbors, (*neighbors[1:], neighbors[0])))
            clears_a = (corner_a[0] & corner_a[1] & corner_a[2]) == 0
            clears_b = (corner_b[0] & corner_b[1] & corner_b[2]) == 0
            peel = (core > 0) & (count >= 2) & (count <= 6) & (transitions == 1) & clears_a & clears_b
            if peel.any():
                core[peel] = 0
                removed = True
        if not removed:
            return core.copy()


def _sample_ink(ink, points):
    x, y = np.moveaxis(np.rint(points).astype(np.int32), -1, 0)
    height, width = ink.shape
    inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    return ink[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)] & inside


def solid_stroke_at_opening(ink, run, position, width):
    """Check whether a proposed opening crosses an unchanged solid wall stroke."""
    direction = run.direction
    normal = np.array([-direction[1], direction[0]])
    center = run.start + ((position - run.start) @ direction) * direction
    radius = min(12, max(4, round(run.thickness / 2 + 3)))
    offsets = np.arange(-radius, radius + 1)[None, :, None] * normal

    def cross_sections(start, end):
        points = center + np.linspace(start, end, 19)[:, None, None] * width * direction + offsets
        return _sample_ink(ink, points).sum(axis=1)

    opening = cross_sections(-.36, .36)
    before = cross_sections(-.75, -.55)
    after = cross_sections(.55, .75)
    solid_width = min(np.median(before), np.median(after))
    return solid_width > 0 and np.mean(opening >= max(1, .8 * solid_width)) >= .85


def trace_wall_runs(structure, scale, source_ink=None):
    """Fit dominant line segments to the supported skeleton and close local junctions."""
    skeleton = _thin(structure)
    minimum_length = max(7, round(.18 / scale))
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=7,
                            minLineLength=minimum_length, maxLineGap=4)
    if lines is None:
        return []

    distance = cv2.distanceTransform(structure, cv2.DIST_L2, 5)
    gap = min(10, max(5, round(.22 / scale)))
    nearby_ink = cv2.dilate(source_ink, np.ones((7, 7), np.uint8)) if source_ink is not None else None
    # A thin strip of nearby parallel evidence lets an axis cross raster stair-steps.
    rails = [
        cv2.dilate(cv2.morphologyEx(structure, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8)), np.ones((7, 1), np.uint8)),
        cv2.dilate(cv2.morphologyEx(structure, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8)), np.ones((1, 7), np.uint8)),
    ]
    candidates = []
    for line in lines.reshape(-1, 4):
        start, end = line.reshape(2, 2).astype(float)
        delta = end - start
        axis = int(np.argmax(np.abs(delta)))
        if abs(delta[1 - axis]) <= min(5, abs(delta[axis]) * .09):
            start[1 - axis] = end[1 - axis] = (start[1 - axis] + end[1 - axis]) / 2
        points = np.rint(np.linspace(start, end, max(5, round(np.linalg.norm(delta))))).astype(int)
        samples = distance[points[:, 1], points[:, 0]]
        if np.count_nonzero(samples) < .65 * len(samples):
            continue
        if nearby_ink is not None and np.mean(_sample_ink(nearby_ink, points)) < .35:
            continue
        thickness = float(np.clip(2 * np.median(samples[samples > 0]) - .5, 2, max(2, .60 / scale)))
        if start[1 - axis] == end[1 - axis]:
            rail = rails[axis][round(start[1])] if axis == 0 else rails[axis][:, round(start[0])]
            rail = cv2.morphologyEx(rail.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, gap // 2 * 2 + 1), np.uint8))[0]
            stops = np.flatnonzero(rail == 0)
            low, high = sorted([start[axis], end[axis]])
            before, after = stops[stops < low], stops[stops > high]
            start[axis] = before[-1] + 1 if before.size else 0
            end[axis] = after[0] - 1 if after.size else len(rail) - 1
        candidates.append(WallRun(start, end, thickness))

    runs = []
    for candidate in sorted(candidates, key=lambda run: run.length, reverse=True):
        for run in runs:
            direction = run.direction
            if abs(direction @ candidate.direction) < .997:
                continue
            offsets = np.array([candidate.start, candidate.end]) - run.start
            normal = np.array([-direction[1], direction[0]])
            if np.max(np.abs(offsets @ normal)) > min(4.5, max(2, run.thickness * .55)):
                continue
            low, high = sorted(offsets @ direction)
            if low > run.length + gap or high < -gap:
                continue
            run.end = run.start + direction * max(run.length, high)
            run.start = run.start + direction * min(0, low)
            break
        else:
            runs.append(candidate)

    # Extend axes to nearby intersections so T and L joints meet before extrusion.
    joins = []
    for i, run in enumerate(runs):
        for j in range(i + 1, len(runs)):
            other = runs[j]
            basis = np.column_stack([run.direction, -other.direction])
            if abs(np.linalg.det(basis)) < .25:
                continue
            a, b = np.linalg.solve(basis, other.start - run.start)
            limit = min(8, max(3, (run.thickness + other.thickness) / 2 + 1))
            extension = max(-a, a - run.length, -b, b - other.length)
            point = run.start + run.direction * a
            if extension > limit:
                if nearby_ink is None or extension > min(17, .45 / scale):
                    continue
                nearest = [wall.start + wall.direction * np.clip(along, 0, wall.length)
                           for wall, along in ((run, a), (other, b))]
                if any(np.mean(_sample_ink(nearby_ink, np.linspace(end, point, 9))) < .75 for end in nearest):
                    continue
            joins.append((i, j, point))

    for i, j, point in joins:
        for run in (runs[i], runs[j]):
            along = (point - run.start) @ run.direction
            if along < 0:
                run.start = point
            elif along > run.length:
                run.end = point
    return runs
