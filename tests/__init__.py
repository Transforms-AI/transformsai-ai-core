"""Test package marker.

Required: ultralytics installs a top-level `tests` package into site-packages, and a
regular package beats a namespace directory during import resolution — without this file
`python -m unittest tests.<name>` picks up ultralytics' tests instead of ours.
"""
