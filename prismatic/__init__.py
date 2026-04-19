# Lazy-load the training loaders so that `import prismatic` (triggered indirectly
# when the inference HTTP server imports `prismatic.extern.hf.*`) does not pull in
# the full RLDS/OXE training pipeline (tensorflow_graphics, dlimp, …). Users who
# still want `prismatic.load` / `prismatic.available_models` get them on demand.


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
