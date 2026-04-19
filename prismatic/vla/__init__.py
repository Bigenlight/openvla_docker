# Lazy-load the training-only `materialize` helper so that importing submodules
# like `prismatic.vla.constants` does not drag in dlimp / RLDS at inference time.


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator  # noqa: F401

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
