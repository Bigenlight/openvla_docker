# Lazy-load the RLDS dataset builders — same rationale as the parent package.


def __getattr__(name):
    if name in {"make_interleaved_dataset", "make_single_dataset"}:
        from .dataset import make_interleaved_dataset, make_single_dataset  # noqa: F401
        return {"make_interleaved_dataset": make_interleaved_dataset,
                "make_single_dataset": make_single_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
