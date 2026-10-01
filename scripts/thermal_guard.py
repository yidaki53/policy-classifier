"""Thermal guard for long-running, GPU-backed batch jobs.

Laptops throttle hard and shut down when the chassis exceeds its thermal
limit. Fixed pauses between batches are a poor control: they cannot react to
an actual temperature spike, and they keep working when the machine is already
cool. This module pauses *until the machine is cool enough to continue*.

Notes on sensor availability:
  * GPU temperature is read through NVML (nvidia-smi). If the NVIDIA kernel
    driver and userspace library are mismatched, NVML fails to initialise and
    no GPU sensor is available, so the guard falls back to CPU and chassis
    sensors only. That is a real limitation, not something this module can
    work around.
  * All sensors are optional. A machine that exposes none simply never pauses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

HWMON_ROOT = Path("/sys/class/hwmon")

# Chassis zones to watch. acpitz tracks the laptop's thermal policy zone and is
# the earliest signal that the case is running hot; coretemp is the CPU.
CHASSIS_SENSORS = ("acpitz",)
CPU_SENSORS = ("coretemp",)

# Conservative defaults for a laptop. The chassis zone trips first, so the
# threshold is well below the CPU's.
DEFAULT_COOL_C = 72.0
DEFAULT_HOT_C = 84.0


def _sensor_paths() -> list[Path]:
    if not HWMON_ROOT.is_dir():
        return []
    paths = []
    for hwmon in sorted(HWMON_ROOT.iterdir()):
        name_file = hwmon / "name"
        temp_file = hwmon / "temp1_input"
        if not (name_file.is_file() and temp_file.is_file()):
            continue
        try:
            name = name_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if name in CHASSIS_SENSORS or name in CPU_SENSORS:
            paths.append(temp_file)
    return paths


def read_temps() -> list[float]:
    """Return readable temperatures in degrees Celsius, or [] if none."""
    values: list[float] = []
    for path in _sensor_paths():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            values.append(int(raw) / 1000.0)
        except (OSError, ValueError):
            continue
    return values


def max_temp() -> Optional[float]:
    """Highest readable temperature, or None when no sensor is available."""
    values = read_temps()
    return max(values) if values else None


def wait_until_cool(
    cool_c: float = DEFAULT_COOL_C,
    hot_c: float = DEFAULT_HOT_C,
    poll_seconds: float = 10.0,
    max_wait_seconds: float = 900.0,
    verbose: bool = True,
) -> bool:
    """Sleep until every readable sensor is at or below cool_c.

    Returns True if the machine was hot and a pause was taken, False if it was
    already cool (or no sensor is readable). Gives up after max_wait_seconds so
    a stuck sensor cannot hang a multi-hour run forever.
    """
    current = max_temp()
    if current is None or current <= cool_c:
        return False

    if verbose:
        print(f"Thermal pause: {current:.1f} C >= {hot_c:.1f} C, waiting for <= {cool_c:.1f} C")

    started = time.monotonic()
    while time.monotonic() - started < max_wait_seconds:
        time.sleep(poll_seconds)
        current = max_temp()
        if current is None or current <= cool_c:
            if verbose:
                cooled = f"{current:.1f} C" if current is not None else "no sensor"
                print(f"Thermal pause over: {cooled}")
            return True
    if verbose:
        print(f"Thermal pause timed out after {max_wait_seconds:.0f}s; resuming anyway")
    return True


class ThermalGuard:
    """Pauses a batch loop when the machine gets too hot.

    Checked every `check_every` items so a hot spot is caught within seconds
    rather than only at the end of a long batch.
    """

    def __init__(
        self,
        check_every: int = 250,
        cool_c: float = DEFAULT_COOL_C,
        hot_c: float = DEFAULT_HOT_C,
        poll_seconds: float = 10.0,
        max_wait_seconds: float = 900.0,
        verbose: bool = True,
    ) -> None:
        self.check_every = max(1, check_every)
        self.cool_c = cool_c
        self.hot_c = hot_c
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.verbose = verbose
        self.pause_count = 0
        self.max_observed_c: Optional[float] = None
        self._counter = 0

    def tick(self) -> bool:
        """Call once per item. Returns True if a thermal pause was taken."""
        self._counter += 1
        if self._counter % self.check_every:
            return False

        current = max_temp()
        if current is not None:
            if self.max_observed_c is None or current > self.max_observed_c:
                self.max_observed_c = current

        if current is None or current < self.hot_c:
            return False

        self.pause_count += 1
        wait_until_cool(
            cool_c=self.cool_c,
            hot_c=self.hot_c,
            poll_seconds=self.poll_seconds,
            max_wait_seconds=self.max_wait_seconds,
            verbose=self.verbose,
        )
        return True


def iter_with_thermal_guard(
    items: Iterable, guard: ThermalGuard
) -> Iterable:
    """Yield items, invoking guard.tick() once per item."""
    for item in items:
        yield item
        guard.tick()