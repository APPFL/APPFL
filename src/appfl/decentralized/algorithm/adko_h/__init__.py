"""ADKO-H: a language model plans campaigns, a calibrated rule picks every experiment.

ADKO v2's settled method with one change of level. In v2 (and in the L3 screen) the model was
offered the pick itself and lost to the score; here it is offered the *design* -- a region of
the space, one of five numeric rules, a budget, and a test it can fail -- and every experiment
inside that design is still the rule's argmax. The campaign's verdict is computed from the data
and travels to neighbours, so coordination happens by message rather than by an allocator and
the method still makes sense at hundreds of agents on a sparse graph.

Entry points::

    from appfl.decentralized.algorithm.adko_h import AgentConfig, RunConfig, run
    from appfl.decentralized.algorithm.adko_h.problems.synthetic import Synthetic
    from appfl.decentralized.algorithm.adko_h.surrogate import GP

    record = run(Synthetic(), GP, RunConfig(rounds=10, agents=4, planner="random"))
"""

from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent, AgentConfig
from appfl.decentralized.algorithm.adko_h.campaign import Campaign, Evidence, Test, random_design
from appfl.decentralized.algorithm.adko_h.claims import Claim, ClaimStore, Verdict
from appfl.decentralized.algorithm.adko_h.loop import RandomPlanner, RunConfig, run
from appfl.decentralized.algorithm.adko_h.problem import Candidates, Outcome, Pick, Problem
from appfl.decentralized.algorithm.adko_h.region import Dim, Region, RegionSpace
from appfl.decentralized.algorithm.adko_h.surrogate import GP
from appfl.decentralized.algorithm.adko_h.token import HToken

__all__ = [
    "ADKOHAgent",
    "AgentConfig",
    "Campaign",
    "Candidates",
    "Claim",
    "ClaimStore",
    "Dim",
    "Evidence",
    "GP",
    "HToken",
    "Outcome",
    "Pick",
    "Problem",
    "RandomPlanner",
    "Region",
    "RegionSpace",
    "RunConfig",
    "Test",
    "Verdict",
    "random_design",
    "run",
]
