# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2) /
# [`Ask_IDA_CLI`](https://github.com/The-Hidden-Canopy/Ask_IDA_CLI). Used with permission.
"""Vendored Omni reference inference harness + minimal model closure.

Self-contained per the OMNI-Q house rule (no cross-repo imports): the
`OmniInference`/`load_omni_reference` harness from Ask_IDA_CLI plus the
`ida_lattice` module subset its checkpoint loader transitively needs. See
SOURCE.md for provenance and the trim/re-wire record.
"""
from .omni_inference import (
    OmniHarnessError,
    OmniInference,
    OmniReply,
    load_omni_reference,
)

__all__ = [
    "OmniHarnessError",
    "OmniInference",
    "OmniReply",
    "load_omni_reference",
]
