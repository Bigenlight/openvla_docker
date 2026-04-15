# The eager `from .models import ...` statement used to live here pulls in the
# full RLDS/OXE training data pipeline (tensorflow_graphics, dlimp, …) just to
# load the pretraining loaders. For inference-only deployments (e.g. the
# FastAPI HTTP server in scripts/serve_openvla_http.py) that chain is dead
# weight and forces a much heavier container. We lazy-import via a module
# `__getattr__` so that `import prismatic` stays cheap, and anyone who still
# wants `prismatic.load` / `prismatic.available_models` picks it up on demand.


def __getattr__(name):
    if name in {"available_model_names", "available_models", "get_model_description", "load"}:
        from .models import (  # noqa: F401
            available_model_names,
            available_models,
            get_model_description,
            load,
        )
        return {
            "available_model_names": available_model_names,
            "available_models": available_models,
            "get_model_description": get_model_description,
            "load": load,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
