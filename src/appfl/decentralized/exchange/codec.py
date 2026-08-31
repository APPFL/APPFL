"""Turning tokens into wire bytes and back."""

from __future__ import annotations

import base64
from typing import List, Sequence

from appfl.decentralized.protocol import TokenProtocol


def default_token_cls() -> type:
    """ADKO's token, imported lazily so this module has no import-time dependency on it.

    A default rather than a required argument because every current caller wants ADKO, and
    forcing four example scripts to pass the same class would be ceremony. The seam is real
    either way: nothing above this function references the class.
    """
    from appfl.decentralized.algorithm.adko.knowledge_token import KnowledgeToken

    return KnowledgeToken


def pack_tokens(tokens: Sequence[TokenProtocol]) -> List[str]:
    """Tokens -> base64 strings, safe to carry in YAML/JSON metadata fields.

    Needs only ``serialize()``, so it is agnostic to what kind of token this is.
    """
    return [base64.b64encode(t.serialize()).decode("ascii") for t in tokens]


def unpack_tokens(packed: Sequence[str], token_cls: type) -> List[TokenProtocol]:
    """Inverse of :func:`pack_tokens`. Malformed entries raise rather than being skipped.

    ``token_cls`` is passed in rather than imported because decoding is the one operation
    that genuinely needs a concrete class -- you cannot reconstruct an object from bytes
    without knowing what to build. Injecting it is what keeps this module free of any
    dependency on a particular algorithm.
    """
    return [token_cls.deserialize(base64.b64decode(p)) for p in packed]
