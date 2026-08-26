from __future__ import annotations

import json

from build_regrid_from_athdf import load_audit_selection


def test_load_stage_s_audit_selection(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "summary": {"raw_merge_gate": "passed"},
                "records": [
                    {"file": "a.athdf", "physical_time": 1.0},
                    {"file": "b.athdf", "physical_time": 2.0},
                ],
            }
        )
    )
    included, excluded, times = load_audit_selection(path)
    assert included == {"a.athdf", "b.athdf"}
    assert excluded == {}
    assert times == {"a.athdf": 1.0, "b.athdf": 2.0}
