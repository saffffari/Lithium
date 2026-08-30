from pathlib import Path

import numpy as np

from src.utils.file_hash import compute_file_key
from tools.migrate_deepfield_ptv3_workspace import (
    ENDPLATE3,
    FULL6,
    P_ENDPLATE3,
    P_GOLD6,
    _compute_file_key,
    _extract_config_names,
    _model_target,
    _ontology,
    _ontology_names,
    _project_id,
    _remap,
)


def test_file_key_matches_application_contract(tmp_path: Path):
    cloud = tmp_path / "bone.ply"
    cloud.write_bytes(b"ply\nend_header\n")
    assert _compute_file_key(cloud) == compute_file_key(str(cloud))


def test_project_ids_are_stable_and_namespaced():
    assert _project_id(P_GOLD6) == _project_id(P_GOLD6)
    assert _project_id(P_GOLD6).startswith("proj:")
    assert _project_id(P_GOLD6) != _project_id(P_ENDPLATE3)


def test_ontology_preserves_output_channel_zero_semantics():
    ontology = _ontology(ENDPLATE3)
    assert _ontology_names(ontology) == ENDPLATE3
    assert ontology["labels"][0]["name"] == "Rest"
    assert ontology["labels"][0]["id"] == 0


def test_known_model_taxonomies_route_to_compatible_projects():
    assert _model_target(FULL6) == (P_GOLD6, "")
    assert _model_target(ENDPLATE3) == (P_ENDPLATE3, "")


def test_unknown_anatomical_channel_zero_is_archive_only():
    target, reason = _model_target(
        ("Superior_Endplate", "Inferior_Endplate", "Process_Tips")
    )
    assert "Model Archive" in target
    assert "channel 0 is anatomical" in reason


def test_label_remaps_do_not_mutate_source():
    source = {"a": np.array([0, 1, 2, 5], dtype=np.int32)}
    mapped = _remap(source, lambda array: np.where(array == 5, 0, array))
    assert source["a"].tolist() == [0, 1, 2, 5]
    assert mapped["a"].tolist() == [0, 1, 2, 0]


def test_config_class_map_is_recovered_from_training_snapshot(tmp_path: Path):
    config = tmp_path / "config.py"
    config.write_text(
        "data = dict(\n"
        "  num_classes=3,\n"
        "  names=['Rest', 'Endplate', 'BodyWall'],\n"
        "  train=dict(type='LithiumDataset'),\n"
        ")\n"
    )
    assert _extract_config_names(config) == ENDPLATE3
