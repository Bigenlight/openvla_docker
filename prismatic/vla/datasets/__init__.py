# Lazy-load training dataset classes so inference-only imports of
# `prismatic.vla.datasets.rlds.utils.data_utils.NormalizationType` don't trigger
# the full dataset pipeline.


def __getattr__(name):
    if name in {"DummyDataset", "EpisodicRLDSDataset", "RLDSBatchTransform", "RLDSDataset"}:
        from .datasets import DummyDataset, EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset  # noqa: F401
        return {
            "DummyDataset": DummyDataset,
            "EpisodicRLDSDataset": EpisodicRLDSDataset,
            "RLDSBatchTransform": RLDSBatchTransform,
            "RLDSDataset": RLDSDataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
