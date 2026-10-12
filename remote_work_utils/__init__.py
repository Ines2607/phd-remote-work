"""remote_work_utils: home, work, remote-work and third-place detection from GPS dwells.

Main entry points::

    from remote_work_utils.settings import load_settings
    from remote_work_utils.pipeline import run_month

    settings = load_settings("configs/demo.json")
    result = run_month(settings, "202004")
    result.user_remote_work  # one row per user, metric flag_remote_work_smoothed_max_thresh
"""

__version__ = "0.1.0"
