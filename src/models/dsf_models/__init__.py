# src/models/__init__.py
# Backwards-compatible exports / aliases for older import names used by scripts.
# Do not rename classes in their own files; just create aliases here.

# import concrete implementations from the real modules
# these modules exist in repo: decision_spikeformer_pssa.py, decision_spikeformer_tssa.py, decision_transformer.py, snn_dt.py (if present)

try:
    from .decision_spikeformer_pssa import SpikeDecisionTransformer, PSSADecisionSpikeFormer
except Exception:
    # swallow import error to allow partial usage; helpful during incremental debugging
    SpikeDecisionTransformer = None
    PSSADecisionSpikeFormer = None

try:
    from .decision_spikeformer_tssa import TSSADecisionSpikeFormer
except Exception:
    TSSADecisionSpikeFormer = None

try:
    from .decision_transformer import DecisionTransformer
except Exception:
    DecisionTransformer = None

# Provide the names older scripts expect by aliasing to actual classes where possible.
# SNNDecisionTransformer was the old name — map it to SpikeDecisionTransformer
SNNDecisionTransformer = SpikeDecisionTransformer

# DecisionSpikeFormer may have referred to either the PSSA or TSSA variant.
# We'll make DecisionSpikeFormer point to the PSSA variant by default (replaceable).
DecisionSpikeFormer = PSSADecisionSpikeFormer

# Also export the concrete names for clarity
__all__ = [
    "SpikeDecisionTransformer",
    "PSSADecisionSpikeFormer",
    "TSSADecisionSpikeFormer",
    "DecisionTransformer",
    "SNNDecisionTransformer",
    "DecisionSpikeFormer",
]
