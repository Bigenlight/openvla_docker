# Lazy-load training utilities. `prismatic/extern/hf/modeling_prismatic.py`
# does `from prismatic.training.train_utils import ...` for inference-safe
# helpers; we don't want that to eagerly import the FSDP/wandb training stack.


def __getattr__(name):
    if name == "get_train_strategy":
        from .materialize import get_train_strategy  # noqa: F401
        return get_train_strategy
    if name in {"Metrics", "VLAMetrics"}:
        from .metrics import Metrics, VLAMetrics  # noqa: F401
        return {"Metrics": Metrics, "VLAMetrics": VLAMetrics}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
