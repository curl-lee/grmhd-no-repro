from grmhd.stage_v_analysis import (
    distribution_shift_error_coupling,
    stage_v_primary_decision,
)


def candidate(**updates):
    value = {
        "delta_shell_skill_vs_stage_t": 0.2,
        "delta_radial_skill_vs_stage_t": 0.2,
        "first_10x_step": 2,
        "O1_state_retention": True,
        "shell_skill": -0.5,
        "radial_skill": -0.2,
    }
    value.update(updates)
    return value


def test_distribution_coupling_contract() -> None:
    assert distribution_shift_error_coupling([0.7, -0.8, 0.1, 0.2]) == "STRONG"
    assert distribution_shift_error_coupling([0.4, 0.35, 0.1, 0.2]) == "MODERATE"
    assert distribution_shift_error_coupling([0.1, 0.2, 0.1, 0.2]) == "LOW"


def test_primary_decision_contract() -> None:
    assert stage_v_primary_decision(
        [candidate(shell_skill=0.1, radial_skill=0.1, first_10x_step=5)],
        gradient_conflict="MODERATE", pairwise_misalignment=False,
    ) == "A"
    assert stage_v_primary_decision(
        [candidate()], gradient_conflict="MODERATE", pairwise_misalignment=False,
    ) == "B"
    assert stage_v_primary_decision(
        [candidate(delta_shell_skill_vs_stage_t=0.0, delta_radial_skill_vs_stage_t=0.0, first_10x_step=1)],
        gradient_conflict="NONE", pairwise_misalignment=False,
    ) == "C"
    assert stage_v_primary_decision(
        [candidate(delta_shell_skill_vs_stage_t=0.0, delta_radial_skill_vs_stage_t=0.0, first_10x_step=1, O1_state_retention=False)],
        gradient_conflict="STRONG", pairwise_misalignment=False,
    ) == "D"
