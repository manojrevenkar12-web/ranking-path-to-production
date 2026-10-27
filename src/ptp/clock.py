"""Wall-clock time, overridable for the simulated seasons (PTP_NOW, seconds since the epoch)."""

from __future__ import annotations

import os
import time


def now() -> int:
    fake = os.environ.get("PTP_NOW")
    return int(float(fake)) if fake else int(time.time())
