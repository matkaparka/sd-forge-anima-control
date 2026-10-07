"""Forge Neo extension installer: the OpenPose mode (DWPose estimator used to train the Pose adapter) needs rtmlib."""
import launch

if not launch.is_installed("rtmlib"):
    launch.run_pip("install rtmlib", "rtmlib (Anima Control, OpenPose mode: whole-body pose detector)")
