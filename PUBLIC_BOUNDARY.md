# OMNI-Q public boundary

This repository is a public, standalone OMNI-Q foundation. Its public surface
contains the capability graph, deterministic planner, bounded local adapters,
simulation contracts, and reproducible synthetic evidence.

The following do not belong in this repository:

- private model bodies, tokenizers, checkpoints, or training artifacts;
- private training or evaluation engines;
- copied source from private repositories;
- private service names, repository URLs, workstation paths, credentials, or
  deployment bindings;
- claims that require an unavailable private runtime.

The optional model-reasoner seam is deliberately fail-closed. A deployment
that has authority to use a private runtime must provide it through
`OMNIQ_REASONER_RUNTIME_ROOT` and, when needed,
`OMNIQ_REASONER_TOKENIZER`. The public checkout never discovers sibling
repositories, infers private paths, or ships the runtime itself.

The public repository may document a protocol or a generic capability, but it
must not publish the implementation, artifact, or operational identity of a
private provider. Evidence in this tree must remain reproducible from the
public checkout or be labelled as an external artifact without embedding its
private source path.
