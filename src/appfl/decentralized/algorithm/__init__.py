"""Decentralized algorithms built on this package's transport, graph, and round driver.

Each subpackage is one algorithm. They share everything below them -- ``exchange``,
``topology``, ``runner``, ``budget``, ``metrics`` -- and know nothing about each other, which
is the property the layering exists to preserve: a new method is a new sibling here, not a
change anywhere else.

Import an algorithm explicitly rather than through the parent package, so a dependency on one
is visible at the import line::

    from appfl.decentralized.algorithm.adko import ADKOAgent

Currently implemented:

* :mod:`~appfl.decentralized.algorithm.adko` -- Agentic Decentralized Knowledge Optimization
  (Rillo et al., arXiv:2605.07863). Decentralized Bayesian optimization in which agents hold
  private surrogates and exchange one compact token per neighbour per round.
* :mod:`~appfl.decentralized.algorithm.adko_v2` -- ADKO v2, the method the v2 study settled
  on: 8-bit robust-score tokens with a best-so-far bit, a 3-round FIFO memory, odds-weighted
  trust and a ``match_rms``-calibrated peer field, plus the optional L3 language-model
  planner. Reproduces the v2 reference implementation bit for bit.
"""

__all__ = []
