"""Typed, repeatable audits over raw player-visible trajectory evidence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Required, TypedDict, cast

from dcss_rl.observation import (
    MenuChoiceApplicability,
    VisibleActionFeedback,
    menu_choice_applicability,
    plain_text,
    visible_action_feedback,
)
from dcss_rl.schema import ActionData, JsonObject, RawMessageData
from dcss_rl.units import AffordanceEventCount, CrawlLogCount, TrajectoryCount

_BERSERK_KEYCODE = ord("a")
_RENOUNCE_KEYCODE = ord("X")


class _TrajectoryRecord(TypedDict, total=False):
    """Fields consumed from the append-only JSON trajectory boundary."""

    type: Required[str]
    action: ActionData
    raw_messages: list[RawMessageData]


@dataclass(frozen=True, slots=True)
class AffordanceAudit:
    """Evidence counts for the current multi-step Berserk research seam."""

    trajectories: TrajectoryCount
    crawl_logs: CrawlLogCount
    ability_menu_opens: AffordanceEventCount
    applicable_berserk_menus: AffordanceEventCount
    inapplicable_berserk_menus: AffordanceEventCount
    missing_berserk_menus: AffordanceEventCount
    berserk_selections: AffordanceEventCount
    renounce_selections: AffordanceEventCount
    ability_menu_cancels: AffordanceEventCount
    berserk_started: AffordanceEventCount
    blocked_active: AffordanceEventCount
    blocked_cooldown: AffordanceEventCount
    stochastic_failures: AffordanceEventCount
    ability_lost: AffordanceEventCount
    zero_turn_blocks: AffordanceEventCount
    renounce_prompt_renderings: AffordanceEventCount
    successful_renouncements: AffordanceEventCount
    wrath_created_deaths: AffordanceEventCount

    def format(self) -> str:
        """Render stable line-oriented output suitable for experiment journals."""
        fields = (
            ("trajectories", self.trajectories),
            ("crawl_logs", self.crawl_logs),
            ("ability_menu_opens", self.ability_menu_opens),
            ("berserk_menu_applicable", self.applicable_berserk_menus),
            ("berserk_menu_inapplicable", self.inapplicable_berserk_menus),
            ("berserk_menu_missing", self.missing_berserk_menus),
            ("berserk_selections", self.berserk_selections),
            ("renounce_selections", self.renounce_selections),
            ("ability_menu_cancels", self.ability_menu_cancels),
            ("berserk_started", self.berserk_started),
            ("blocked_active", self.blocked_active),
            ("blocked_cooldown", self.blocked_cooldown),
            ("stochastic_failures", self.stochastic_failures),
            ("ability_lost", self.ability_lost),
            ("zero_turn_blocks", self.zero_turn_blocks),
            ("renounce_prompt_renderings", self.renounce_prompt_renderings),
            ("successful_renouncements", self.successful_renouncements),
            ("wrath_created_deaths", self.wrath_created_deaths),
        )
        return "\n".join(f"{name}: {value}" for name, value in fields)


def audit_affordances(inputs: tuple[Path, ...]) -> AffordanceAudit:
    """Audit files or recursive roots without relying on stored semantic reductions."""
    trajectories, crawl_logs = _transcript_paths(inputs)
    counts = _MutableAudit()
    for trajectory in trajectories:
        ability_menu_visible = False
        for record in _records(trajectory):
            if record["type"] != "transition":
                continue
            action = record.get("action")
            if action is not None:
                if action["kind"] == "menu_select" and ability_menu_visible:
                    if action["keycode"] == _BERSERK_KEYCODE:
                        counts.berserk_selections += 1
                    elif action["keycode"] == _RENOUNCE_KEYCODE:
                        counts.renounce_selections += 1
                elif action["kind"] == "cancel" and ability_menu_visible:
                    counts.ability_menu_cancels += 1

            raw_messages = record.get("raw_messages", [])
            feedback = visible_action_feedback(_visible_messages(raw_messages))
            counts.add_feedback(feedback)
            if (
                VisibleActionFeedback.BERSERK_ACTIVE_REJECTED in feedback
                or VisibleActionFeedback.BERSERK_COOLDOWN_REJECTED in feedback
            ) and not _has_player_turn_update(raw_messages):
                counts.zero_turn_blocks += 1

            ability_menu_visible = _update_menu_state(
                ability_menu_visible, raw_messages, counts
            )
    for crawl_log in crawl_logs:
        _audit_crawl_log(crawl_log, counts)
    return counts.freeze(
        TrajectoryCount(len(trajectories)), CrawlLogCount(len(crawl_logs))
    )


@dataclass(slots=True)
class _MutableAudit:
    ability_menu_opens: int = 0
    applicable_berserk_menus: int = 0
    inapplicable_berserk_menus: int = 0
    missing_berserk_menus: int = 0
    berserk_selections: int = 0
    renounce_selections: int = 0
    ability_menu_cancels: int = 0
    berserk_started: int = 0
    blocked_active: int = 0
    blocked_cooldown: int = 0
    stochastic_failures: int = 0
    ability_lost: int = 0
    zero_turn_blocks: int = 0
    renounce_prompt_renderings: int = 0
    successful_renouncements: int = 0
    wrath_created_deaths: int = 0

    def add_feedback(self, feedback: frozenset[VisibleActionFeedback]) -> None:
        self.berserk_started += VisibleActionFeedback.BERSERK_STARTED in feedback
        self.blocked_active += VisibleActionFeedback.BERSERK_ACTIVE_REJECTED in feedback
        self.blocked_cooldown += (
            VisibleActionFeedback.BERSERK_COOLDOWN_REJECTED in feedback
        )
        self.stochastic_failures += VisibleActionFeedback.ABILITY_FAILED in feedback
        self.ability_lost += VisibleActionFeedback.ABILITY_LOST in feedback

    def freeze(
        self, trajectories: TrajectoryCount, crawl_logs: CrawlLogCount
    ) -> AffordanceAudit:
        return AffordanceAudit(
            trajectories,
            crawl_logs,
            *(
                AffordanceEventCount(value)
                for value in (
                    self.ability_menu_opens,
                    self.applicable_berserk_menus,
                    self.inapplicable_berserk_menus,
                    self.missing_berserk_menus,
                    self.berserk_selections,
                    self.renounce_selections,
                    self.ability_menu_cancels,
                    self.berserk_started,
                    self.blocked_active,
                    self.blocked_cooldown,
                    self.stochastic_failures,
                    self.ability_lost,
                    self.zero_turn_blocks,
                    self.renounce_prompt_renderings,
                    self.successful_renouncements,
                    self.wrath_created_deaths,
                )
            ),
        )


def _transcript_paths(
    inputs: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    trajectories: set[Path] = set()
    crawl_logs: set[Path] = set()
    for supplied in inputs:
        if supplied.is_file():
            if supplied.name == "trajectory.jsonl":
                trajectories.add(supplied)
            elif supplied.name == "crawl.log":
                crawl_logs.add(supplied)
            else:
                raise ValueError(f"unsupported transcript file: {supplied}")
        elif supplied.is_dir():
            trajectories.update(supplied.rglob("trajectory.jsonl"))
            crawl_logs.update(supplied.rglob("crawl.log"))
        else:
            raise FileNotFoundError(supplied)
    if not trajectories and not crawl_logs:
        raise ValueError("no trajectory.jsonl or crawl.log files found")
    return tuple(sorted(trajectories)), tuple(sorted(crawl_logs))


def _records(path: Path) -> Iterator[_TrajectoryRecord]:
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        decoded = json.loads(line)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("type"), str):
            raise ValueError(f"invalid trajectory record at {path}:{line_number}")
        yield cast(_TrajectoryRecord, decoded)


def _audit_crawl_log(path: Path, counts: _MutableAudit) -> None:
    text = path.read_bytes().decode(errors="replace")
    counts.renounce_prompt_renderings += text.count("Really renounce your faith")
    if "You have lost your religion!" not in text:
        return
    counts.successful_renouncements += 1
    if any(
        "created by the rage of Trog" in morgue.read_bytes().decode(errors="replace")
        for morgue in (path.parent / "morgue").glob("*.txt")
    ):
        counts.wrath_created_deaths += 1


def _visible_messages(raw_messages: list[RawMessageData]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in raw_messages:
        payload = raw["payload"]
        values = payload.get("messages")
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                result.append(plain_text(value["text"]))
    return tuple(result)


def _has_player_turn_update(raw_messages: list[RawMessageData]) -> bool:
    return any(
        raw["payload"].get("msg") == "player"
        and isinstance(raw["payload"].get("turn"), int)
        for raw in raw_messages
    )


def _update_menu_state(
    current: bool, raw_messages: list[RawMessageData], counts: _MutableAudit
) -> bool:
    visible = current
    for raw in raw_messages:
        payload = raw["payload"]
        kind = payload.get("msg")
        if kind in {"close_menu", "close_all_menus", "ui-pop"}:
            visible = False
        if kind == "menu" and payload.get("tag") == "ability":
            visible = True
            counts.ability_menu_opens += 1
            _count_berserk_menu(payload, counts)
    return visible


def _count_berserk_menu(payload: JsonObject, counts: _MutableAudit) -> None:
    items = payload.get("items")
    if not isinstance(items, list):
        counts.missing_berserk_menus += 1
        return
    berserk = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and "berserk" in plain_text(item["text"]).casefold()
        ),
        None,
    )
    if berserk is None:
        counts.missing_berserk_menus += 1
    elif (
        menu_choice_applicability("ability", berserk)
        is MenuChoiceApplicability.INAPPLICABLE
    ):
        counts.inapplicable_berserk_menus += 1
    else:
        counts.applicable_berserk_menus += 1
