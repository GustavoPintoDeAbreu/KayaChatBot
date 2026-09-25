"""The GPU PC's power schedule, from config.yaml ``power:``.

One block, three consumers: the PC's shutdown timer (deploy/power), the Pi's
Wake-on-LAN timer (scripts/deploy_pi.sh) and the gateway's offline reply. Each
derives what it needs from here rather than restating the times.

``shutdown`` is keyed by the EVENING it belongs to. A time before noon means
after midnight, so ``fri: "02:00"`` fires early on Saturday. That is how people
describe it ("Friday we stay up until two"), and it keeps a late night one entry.

    python -m src.gateway.schedule [--config config.yaml] ACTION
    ACTION: oncalendar-shutdown | oncalendar-wake | next-wake-epoch | next-shutdown | is-on
"""
from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, Iterator, List, Tuple
from zoneinfo import ZoneInfo

import yaml

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Wide enough that a weekly schedule always has a wake and a shutdown on each side.
_WINDOW_DAYS = 8


@dataclass(frozen=True)
class PowerSchedule:
    """When the PC wakes (daily) and when it shuts down (per evening)."""

    timezone: ZoneInfo
    wake_time: datetime.time
    wol_lead: datetime.timedelta
    shutdown: Dict[int, datetime.time]

    @classmethod
    def from_config(cls, power: Dict[str, Any]) -> "PowerSchedule":
        """Build the schedule from the ``power:`` block of config.yaml."""
        shutdown = {
            _DAY_NAMES.index(day.capitalize()[:3]): datetime.time.fromisoformat(str(at))
            for day, at in (power.get("shutdown") or {}).items()
        }
        return cls(
            timezone=ZoneInfo(power.get("timezone", "Europe/Lisbon")),
            wake_time=datetime.time.fromisoformat(str(power.get("wake_time", "07:00"))),
            wol_lead=datetime.timedelta(minutes=float(power.get("wol_lead_minutes", 5))),
            shutdown=shutdown,
        )

    def _local(self, moment: datetime.datetime) -> datetime.datetime:
        """``moment`` in the schedule's timezone; a naive value is taken as local."""
        if moment.tzinfo is None:
            return moment.replace(tzinfo=self.timezone)
        return moment.astimezone(self.timezone)

    def _events(self, around: datetime.datetime) -> Iterator[Tuple[datetime.datetime, bool]]:
        """``(moment, is_wake)`` for every wake and shutdown near ``around``."""
        for offset in range(-_WINDOW_DAYS, _WINDOW_DAYS + 1):
            day = around.date() + datetime.timedelta(days=offset)
            yield datetime.datetime.combine(day, self.wake_time, tzinfo=self.timezone), True
            at = self.shutdown.get(day.weekday())
            if at is not None:
                fire_day = day + datetime.timedelta(days=1) if at.hour < 12 else day
                yield datetime.datetime.combine(fire_day, at, tzinfo=self.timezone), False

    def shutdown_moments(self, around: datetime.datetime) -> List[datetime.datetime]:
        """Every scheduled shutdown within the window around ``around``, sorted."""
        local = self._local(around)
        return sorted(moment for moment, is_wake in self._events(local) if not is_wake)

    def is_scheduled_on(self, moment: datetime.datetime) -> bool:
        """Whether the schedule has the PC on at ``moment`` (last event was a wake)."""
        local = self._local(moment)
        past = [event for event in self._events(local) if event[0] <= local]
        return max(past)[1] if past else True

    def next_wake(self, moment: datetime.datetime) -> datetime.datetime:
        """The first wake strictly after ``moment``."""
        local = self._local(moment)
        return min(at for at, is_wake in self._events(local) if is_wake and at > local)

    def next_shutdown(self, moment: datetime.datetime) -> datetime.datetime:
        """The first shutdown strictly after ``moment``."""
        local = self._local(moment)
        upcoming = [at for at, is_wake in self._events(local) if not is_wake and at > local]
        if not upcoming:
            raise ValueError("no shutdown is scheduled")
        return min(upcoming)

    def shutdown_oncalendar(self) -> List[str]:
        """systemd ``OnCalendar=`` values: e.g. ``["Sat,Sun 02:00", "Mon..Fri 23:00"]``."""
        fired: List[Tuple[str, int]] = []
        for evening, at in self.shutdown.items():
            weekday = (evening + 1) % 7 if at.hour < 12 else evening
            fired.append((at.strftime("%H:%M"), weekday))
        lines = []
        for clock, group in groupby(sorted(fired), key=lambda item: item[0]):
            lines.append(f"{_systemd_days(sorted({weekday for _, weekday in group}))} {clock}")
        return lines

    def wake_oncalendar(self) -> str:
        """systemd ``OnCalendar=`` for the Wake-on-LAN packet: wake time minus the lead."""
        anchor = datetime.datetime.combine(datetime.date(2000, 1, 3), self.wake_time)
        return (anchor - self.wol_lead).strftime("*-*-* %H:%M:%S")


_EVENINGS_PT = ["segundas", "terças", "quartas", "quintas", "sextas", "sábados", "domingos"]


def _join_pt(items: List[str]) -> str:
    """"a", "a e b", "a, b e c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


def schedule_sentence(schedule: PowerSchedule) -> str:
    """The hours in words, for a WhatsApp reply: "das 07:00 às 23:00, e até às 02:00 às sextas e sábados".

    The most common shutdown time is the rule and the others are named as
    exceptions, by the evening they belong to (as people say it).
    """
    wake = schedule.wake_time.strftime("%H:%M")
    if not schedule.shutdown:
        return f"sempre ligado a partir das {wake}"
    by_time: Dict[str, List[int]] = {}
    for evening, at in sorted(schedule.shutdown.items()):
        by_time.setdefault(at.strftime("%H:%M"), []).append(evening)
    usual = max(by_time, key=lambda clock: len(by_time[clock]))
    sentence = f"das {wake} às {usual}"
    for clock, evenings in sorted(by_time.items()):
        if clock != usual:
            sentence += f", e até às {clock} às {_join_pt([_EVENINGS_PT[day] for day in evenings])}"
    return sentence


def _systemd_days(weekdays: List[int]) -> str:
    """Sorted weekday numbers as systemd days: runs of three or more become ``A..B``."""
    runs: List[List[int]] = []
    for weekday in weekdays:
        if runs and weekday == runs[-1][-1] + 1:
            runs[-1].append(weekday)
        else:
            runs.append([weekday])
    parts = []
    for run in runs:
        if len(run) >= 3:
            parts.append(f"{_DAY_NAMES[run[0]]}..{_DAY_NAMES[run[-1]]}")
        else:
            parts.extend(_DAY_NAMES[weekday] for weekday in run)
    return ",".join(parts)


def load_schedule(config_path: str) -> PowerSchedule:
    """The schedule from a config.yaml, read directly (no profile machinery)."""
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return PowerSchedule.from_config(config.get("power") or {})


def main() -> None:
    """Print one schedule value; used by the systemd installers and the shutdown script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("action", choices=["oncalendar-shutdown", "oncalendar-wake",
                                           "next-wake-epoch", "next-shutdown", "is-on"])
    args = parser.parse_args()
    schedule = load_schedule(args.config)
    now = datetime.datetime.now(schedule.timezone)
    if args.action == "oncalendar-shutdown":
        print("\n".join(schedule.shutdown_oncalendar()))
    elif args.action == "oncalendar-wake":
        print(schedule.wake_oncalendar())
    elif args.action == "next-wake-epoch":
        print(int(schedule.next_wake(now).timestamp()))
    elif args.action == "next-shutdown":
        print(schedule.next_shutdown(now).isoformat())
    else:
        print("true" if schedule.is_scheduled_on(now) else "false")


if __name__ == "__main__":
    main()
