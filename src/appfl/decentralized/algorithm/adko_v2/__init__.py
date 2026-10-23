"""ADKO v2 -- the settled method, over APPFL's transport.

The second version of Agentic Decentralized Knowledge Optimization: the method the v2 study
settled on (arm ``ADKO-v2f-odds-bit``), ported from the authors' reference implementation.
One configuration everywhere:

* ``R = mu + 2 sigma + 2 G - 2 Lambda``, with the peer terms rescaled by ``match_rms``;
* 8-bit tokens: my result's robust score against my running median, clipped to +-1;
* one token a round, flagged when it is my new best; receivers keep each peer's newest flagged
  token past FIFO;
* a FIFO memory of 3 rounds;
* trust with odds weights ``t / (1 - t)``, Bush-Mosteller updates from ``t = 1/2``, ``eta = 0.1``;
* ``sigma = 1`` (Euclidean) or ``0.5`` (Hamming).

Optional module: **L3**, language-model proposals (``llm/``, the reference's ``llm_v2`` file
for file). With probability ``alpha`` per agent-round an LLM picks the experiment instead of the
score; its hypothesis travels with that experiment's token. Implemented but not yet validated
against the reference, where it is still changing.

The layering is v1's: everything here is algorithm, and the transport, round driver, graph and
budget one level up know nothing about it. ``../adko`` is the v1 method, kept as it was.
"""

from appfl.decentralized.algorithm.adko_v2.knowledge_token import (
    TAG_BEST,
    TAG_RESULT,
    KnowledgeToken,
    Provenance,
    Quantizer,
    encode_token,
    relative_output,
    running_reference,
    score_against,
)
from appfl.decentralized.algorithm.adko_v2.components import (
    DesignSpace,
    Planner,
    Surrogate,
)
from appfl.decentralized.algorithm.adko_v2.memory import TokenMemory
from appfl.decentralized.algorithm.adko_v2.reasoning import (
    Calibrator,
    PeerField,
    ReasoningWeights,
    field_coefficients,
    score,
    similarity,
    source_weights,
)
from appfl.decentralized.algorithm.adko_v2.trust import (
    TrustState,
    agreement,
    relative_sigma,
)
from appfl.decentralized.algorithm.adko_v2.agent import ADKOv2Agent
from appfl.decentralized.algorithm.adko_v2.metrics import ADKOv2Meter
from appfl.decentralized.algorithm.adko_v2.llm import (
    L3Policy,
    RandomPolicy,
    build_policy,
)
from appfl.decentralized.algorithm.adko_v2.builder import (
    create_agent,
    create_planner,
    create_weights,
)

__all__ = [
    # the token
    "KnowledgeToken",
    "Provenance",
    "Quantizer",
    "TAG_RESULT",
    "TAG_BEST",
    "encode_token",
    "relative_output",
    "running_reference",
    "score_against",
    # interfaces the science owner fills in
    "Surrogate",
    "DesignSpace",
    "Planner",
    # the algorithm
    "ADKOv2Agent",
    "TokenMemory",
    "ReasoningWeights",
    "PeerField",
    "Calibrator",
    "similarity",
    "source_weights",
    "field_coefficients",
    "score",
    "TrustState",
    "agreement",
    "relative_sigma",
    # construction from configuration
    "create_agent",
    "create_weights",
    "create_planner",
    "ADKOv2Meter",
    # the L3 module
    "L3Policy",
    "RandomPolicy",
    "build_policy",
]


def _allow_tokens_through_torch_serialization() -> None:
    """Declare the token types safe for ``torch.load``, exactly as ``adko`` does for its own.

    APPFL's transports serialize payloads with ``torch.save``/``torch.load``, which since
    PyTorch 2.6 refuses any class it has not been told about. These two are plain data -- two
    dataclasses of numbers, strings and lists, with no custom ``__reduce__``.
    """
    try:
        import torch
    except ImportError:
        return
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([KnowledgeToken, Provenance])


_allow_tokens_through_torch_serialization()
