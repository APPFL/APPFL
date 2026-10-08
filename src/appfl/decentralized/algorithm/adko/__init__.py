"""ADKO -- Agentic Decentralized Knowledge Optimization, over APPFL's transport.

Rillo et al., arXiv:2605.07863; reference implementations at ``github.com/lucasrillo/adko``.
Decentralized Bayesian optimization in which agents hold private surrogates and exchange one
compact token per neighbour per round instead of data, models, or gradients.

Everything here is algorithm. The transport it runs over (``exchange``), the round driver
(``runner``), the graph (``topology``) and the budget accounting (``budget``) live one level
up and know nothing about any of this -- which is what lets a different decentralized method
reuse them unchanged.

Start with ``../README.md``: §2-3 for the token and the reasoning score, §9 for where the two
published studies disagree, §12 for the configuration presets.
"""

from appfl.decentralized.algorithm.adko.knowledge_token import (
    KnowledgeToken,
    Provenance,
    Signal,
    binary_entropy,
    encode_token,
)
from appfl.decentralized.algorithm.adko.components import DesignSpace, LanguageModel, Surrogate
from appfl.decentralized.algorithm.adko.baseline import (
    BaseBaseline,
    FixedBaseline,
    RunningMedianBaseline,
    build_baseline,
    get_appfl_baseline,
)
from appfl.decentralized.algorithm.adko.pruning import (
    ConfidencePruner,
    FIFOPruner,
    FidelityAwarePruner,
    RandomPruner,
    TokenPruner,
    merge,
)
from appfl.decentralized.algorithm.adko.reasoning import (
    ReasoningWeights,
    bandwidth_for_dimension,
    distance,
    peer_terms,
    reasoning_score,
    score_candidates,
    similarity,
)
from appfl.decentralized.algorithm.adko.agent import ADKOAgent
from appfl.decentralized.algorithm.adko.metrics import ADKOMeter
from appfl.decentralized.algorithm.adko.llm import (
    LLMConfig,
    OpenAILanguageModel,
    build_language_model,
)

from appfl.decentralized.algorithm.adko.builder import (
    PRUNERS,
    WEIGHT_PRESETS,
    create_agent,
    create_weights,
)

__all__ = [
    # the token
    "KnowledgeToken",
    "Signal",
    "Provenance",
    "encode_token",
    "binary_entropy",
    # interfaces the science owner fills in
    "Surrogate",
    "LanguageModel",
    "DesignSpace",
    # baselines
    "BaseBaseline",
    "FixedBaseline",
    "RunningMedianBaseline",
    "build_baseline",
    "get_appfl_baseline",
    # the algorithm
    "ADKOAgent",
    # construction from configuration
    "create_agent",
    "create_weights",
    "PRUNERS",
    "WEIGHT_PRESETS",
    "ADKOMeter",
    "ReasoningWeights",
    "reasoning_score",
    "score_candidates",
    "peer_terms",
    "similarity",
    "distance",
    "bandwidth_for_dimension",
    "TokenPruner",
    "FidelityAwarePruner",
    "ConfidencePruner",
    "FIFOPruner",
    "RandomPruner",
    "merge",
    # language model
    "LLMConfig",
    "OpenAILanguageModel",
    "build_language_model",
]

def _allow_tokens_through_torch_serialization() -> None:
    """Declare the token types as safe for ``torch.load``.

    APPFL's transports serialize payloads with ``torch.save``/``torch.load``, and since
    PyTorch 2.6 ``torch.load`` defaults to ``weights_only=True`` -- it will refuse any class
    it has not been told about, which is exactly the protection that should stay on for model
    parameters arriving from a peer. These three are plain data: an enum of two strings and
    two dataclasses of floats, strings and lists, with no custom ``__reduce__``. Declaring
    them here, where they are defined, is what lets a knowledge token travel the same wire a
    state dict does without any launcher loosening the check globally.
    """
    try:
        import torch
    except ImportError:
        return
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([KnowledgeToken, Provenance, Signal])

_allow_tokens_through_torch_serialization()
