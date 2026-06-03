from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

from observation_labeler.stages.fallbacks import FallbackRegistry


class _ModelAdapter:
    def __init__(self, model) -> None:  # noqa: ANN001
        self._model = model
        self.model_version = getattr(model, "model_version", "unknown")

    def predict(self, text: str) -> dict[str, float]:
        if not getattr(self._model, "available", False):
            return {}
        return self._model.predict(text)


def build_signals_fallback_registry(
    *,
    models_dir: str,
    signals_repo: str = "/home/krish-agarwal/oxygen/signals",
    enabled_fallbacks: set[str] | None = None,
) -> FallbackRegistry:
    """Use the existing strong model wrappers, but return labels only.

    This imports model wrapper code from the sibling `signals` checkout. The
    observation labeler still does not make guardrail decisions; these adapters
    only expose model scores for fallback labels.
    """

    repo = Path(signals_repo).expanduser()
    wrapper_dir = repo / "agent_observability_sdk" / "gates" / "precallgate" / "models"
    enabled = enabled_fallbacks or {"fallback_a", "fallback_b", "fallback_c", "fallback_d"}
    models = {}

    if "fallback_a" in enabled:
        deberta_module = _load_module(wrapper_dir / "deberta_injection.py", "metric_work_deberta_injection")
        models["fallback_a"] = _ModelAdapter(deberta_module.DebertaInjectionModel(models_dir))
    if "fallback_b" in enabled:
        detoxify_module = _load_module(wrapper_dir / "detoxify_onnx.py", "metric_work_detoxify_onnx")
        models["fallback_b"] = _ModelAdapter(detoxify_module.DetoxifyModel(models_dir))
    if "fallback_c" in enabled:
        jailbreak_module = _load_module(wrapper_dir / "jailbreak_onnx.py", "metric_work_jailbreak_onnx")
        models["fallback_c"] = _ModelAdapter(jailbreak_module.JailbreakModel(models_dir))
    if "fallback_d" in enabled:
        koala_module = _load_module(wrapper_dir / "koala_moderation_onnx.py", "metric_work_koala_moderation_onnx")
        models["fallback_d"] = _ModelAdapter(koala_module.KoalaModerationModel(models_dir))
    return FallbackRegistry(models=models)


def _load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
