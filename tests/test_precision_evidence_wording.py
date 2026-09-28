from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mxfp2_docs_distinguish_target_residency_from_hardware_evidence() -> None:
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    ontology = (ROOT / "docs" / "ontology.md").read_text(encoding="utf-8")

    combined = architecture + "\n" + ontology
    assert "current evidence is software-dequantized" in combined
    assert "native\nMXFP2 execution" in combined
    assert "hardware path remains unverified" in combined
    assert "NVINT2" in ontology and "execution is not yet evidenced" in ontology

    # These were the unsupported present-tense claims that caused the drift.
    assert "runs at **MXFP2** continuously on the Intel box" not in ontology
    assert "run at **MXFP2** for the swarm)" not in architecture
