from pneumoai.models.loader import load_model_bundle


def test_model_bundle_loads():
    model, version, threshold, metadata = load_model_bundle()

    assert model is not None
    assert isinstance(version, str)
    assert isinstance(threshold, float)
    assert isinstance(metadata, dict)