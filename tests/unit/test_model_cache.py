from pneumoai.serving.dispatcher.inference_service import (
    _get_model_bundle,
    clear_model_cache,
)


def test_model_cache_reuses_bundle_for_same_version():
    clear_model_cache()

    first = _get_model_bundle("v2")
    second = _get_model_bundle("v2")

    assert first is second
    assert first["version"] == "v2"
    assert float(first["threshold"]) == 0.45


def test_model_cache_can_be_cleared():
    clear_model_cache()

    first = _get_model_bundle("v2")
    clear_model_cache()
    second = _get_model_bundle("v2")

    assert first is not second
    assert second["version"] == "v2"