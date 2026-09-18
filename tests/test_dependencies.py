

def test_new_deps_importable():
    import nnsight  # noqa: F401
    import nnterp  # noqa: F401
    import torch  # noqa: F401


def test_transformers_recognizes_qwen3():
    # Qwen3 support landed in transformers alongside the model's April 2025 release.
    # This is the actual gate for "is our pinned transformers new enough" -- not a
    # version-string comparison, which would need updating every time transformers
    # renumbers.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    assert "qwen3" in CONFIG_MAPPING
