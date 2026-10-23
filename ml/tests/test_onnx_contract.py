import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
MODEL = ML_ROOT / "models" / "draft_model_xgboost_full.onnx"
FEATURES = ML_ROOT / "models" / "draft_model_xgboost_full_features.json"
SCHEMA = ML_ROOT / "models" / "draft_model_xgboost_full_schema.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_draft_features.json"


@pytest.fixture(scope="module")
def contract():
    feature_names = json.loads(FEATURES.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return feature_names, schema, fixture


def test_schema_matches_features(contract):
    feature_names, schema, _ = contract

    assert schema["feature_count"] == len(feature_names) == 887
    assert schema["feature_names"] == feature_names
    assert schema["positive_class_index"] == 1
    assert schema["probability_output"] == "probabilities"


def test_categorical_contract(contract):
    _, schema, _ = contract

    assert schema["categoricals"]["patch"][-1] == "7.41"
    assert schema["categoricals"]["patch"].index("7.41") == 24
    assert schema["categoricals"]["patch_version"].index("7.41") == 1
    assert schema["defaults"]["patch"] == 24.0
    assert schema["defaults"]["patch_version"] == 1.0
    assert schema["defaults"]["duration"] == 2163.0


def test_golden_vector_matches_onnx(contract):
    feature_names, _, fixture = contract
    assert fixture["feature_names"] == feature_names
    assert len(fixture["input"]) == len(feature_names)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_array = np.asarray([fixture["input"]], dtype=np.float32)
    label, probabilities = session.run(None, {input_name: input_array})

    assert int(label[0]) == fixture["expected"]["label"]
    assert probabilities.shape == (1, 2)
    assert np.allclose(
        probabilities[0],
        fixture["expected"]["probabilities"],
        rtol=0,
        atol=1e-6,
    )
    assert probabilities[0][1] == pytest.approx(
        fixture["expected"]["positive_probability"],
        abs=1e-6,
    )
    assert probabilities[0][0] != probabilities[0][1]
