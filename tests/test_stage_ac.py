import pytest

from grmhd.stage_ac import audit_radial_hat_feasibility


def test_stage_ac_frozen_64_contract_exposes_zero_support_hats():
    result = audit_radial_hat_feasibility(
        in_shape=(64, 64, 64),
        out_shape=(64, 64, 64),
        domain_length=(2.0, 2.0, 2.0),
        kernel_size=5,
    )
    assert result["radius_cutoff"] == pytest.approx(0.03125)
    assert result["local_stencil_shape"] == [3, 3, 3]
    assert result["unique_supported_radii_within_cutoff"] == pytest.approx([0.0, 0.03125])
    assert result["zero_support_basis_indices"] == [1, 2, 3]
    assert result["all_basis_normalizable"] is False
    assert result["quadrature_normalization_contract_pass"] is False


def test_stage_ac_feasibility_audit_rejects_nondivisible_scaling():
    with pytest.raises(ValueError, match="divide"):
        audit_radial_hat_feasibility(
            in_shape=(8, 8, 8),
            out_shape=(5, 5, 5),
            domain_length=(2.0, 2.0, 2.0),
        )
