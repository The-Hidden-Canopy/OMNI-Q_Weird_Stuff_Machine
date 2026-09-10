# Portions derived from *The Hidden Canopy LLC* —
# [`IDA-TRAIN-V2`](https://github.com/The-Hidden-Canopy/IDA-TRAIN-V2). Used with permission.
"""Trimmed `ida_lattice` package for the Omni reference inference path.

Vendored subset of IDA-TRAIN-V2's `src/ida_train/models/ida_lattice/` — only
the modules transitively required by `omni_state_model`
(`OmniMorphableForCausalLM`, reference/torch forward pass). The source repo's
`__init__.py` re-exported the entire lattice family tree
(`IDALatticeForCausalLM`, governed memory, thalamic router, temporal memory,
...); none of that is vendored and this `__init__` deliberately performs no
eager re-exports. Import submodules directly. See ../SOURCE.md.
"""
