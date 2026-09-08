import json
from pathlib import Path

from dcss_rl.audit import audit_affordances


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_audit_counts_raw_ability_flow_and_zero_turn_rejection(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "episode" / "trajectory.jsonl"
    write_jsonl(
        trajectory,
        [
            {"type": "episode"},
            {
                "type": "transition",
                "action": {"kind": "abilities", "keycode": None},
                "raw_messages": [
                    {
                        "control": False,
                        "payload": {
                            "msg": "menu",
                            "tag": "ability",
                            "items": [
                                {
                                    "hotkeys": [97],
                                    "text": " a - Berserk",
                                    "colour": 8,
                                },
                                {"hotkeys": [88], "text": " X - Renounce Religion"},
                            ],
                        },
                    }
                ],
            },
            {
                "type": "transition",
                "action": {"kind": "menu_select", "keycode": 97},
                "raw_messages": [
                    {
                        "control": False,
                        "payload": {
                            "msg": "msgs",
                            "messages": [{"text": "You are too berserk!"}],
                        },
                    }
                ],
            },
        ],
    )

    report = audit_affordances((tmp_path,))

    assert report.trajectories == 1
    assert report.ability_menu_opens == 1
    assert report.inapplicable_berserk_menus == 1
    assert report.berserk_selections == 1
    assert report.blocked_active == 1
    assert report.zero_turn_blocks == 1


def test_audit_counts_training_log_renunciation_and_wrath_death(
    tmp_path: Path,
) -> None:
    run = tmp_path / "worker-1"
    run.mkdir()
    (run / "crawl.log").write_text(
        "Really renounce your faith?\nReally renounce your faith?\n"
        "You have lost your religion!\n"
    )
    morgue = run / "morgue" / "morgue-test.txt"
    morgue.parent.mkdir()
    morgue.write_text("Slain by a polar bear created by the rage of Trog")

    report = audit_affordances((tmp_path,))

    assert report.crawl_logs == 1
    assert report.renounce_prompt_renderings == 2
    assert report.successful_renouncements == 1
    assert report.wrath_created_deaths == 1
