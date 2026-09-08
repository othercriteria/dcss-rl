from dcss_rl.replay_compare import compare_actions
from dcss_rl.units import ActionIndex


def test_inserted_cancel_does_not_count_as_changed_suffix() -> None:
    before = tuple(map(ActionIndex, (0, 1, 2, 3)))
    after = tuple(map(ActionIndex, (0, 1, 13, 2, 3)))
    edits = compare_actions(before, after)
    assert len(edits) == 1
    edit = edits[0]
    assert edit.operation == "insert"
    assert (edit.reference_start, edit.candidate_start) == (2, 2)
    assert edit.reference_actions == ()
    assert edit.candidate_actions == (13,)


def test_long_repetition_keeps_exact_action_comparison() -> None:
    actions = (ActionIndex(0), ActionIndex(1)) * 500
    assert compare_actions(actions, actions) == ()
