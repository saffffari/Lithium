"""Geometry derivation + clinical measurement layer for SpineLab.

The model produces semantic-segmentation predictions; the metrics
layer turns those into anatomical landmarks, vertebral coordinate
frames, and the actual clinical measurements (disc heights, listhesis,
Cobb angles, sagittal alignment etc).

This package is **deterministic numpy + scipy**. No torch, no GPU, no
GUI. Designed to be runnable standalone for batch metric computation
on a labelled dataset (or on prediction outputs).
"""
