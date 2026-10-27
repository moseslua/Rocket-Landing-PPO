"""Run third-party environment checks with `python scripts/smoke.py`."""

from rocket_landing.smoke import main

if __name__ == "__main__":
    raise SystemExit(main([]))
