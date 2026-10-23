"""L3: the optional language-model module of ADKO v2 -- sometimes the LLM picks the experiment.

Each round after warmup, agent ``i`` flips a coin ``u(seed, i, t) < alpha`` (``alpha = 0.3``),
with the same flips in every arm. Heads: its LLM picks a condition the agent may run and has
not run (if invalid: its two fallbacks, then the score). Tails: the score picks. The LLM sees
its lab's results, the reagent names, its earlier picks with their hypotheses and outcomes, and
its peers' tokens, each marked as a new result or a best so far. The text its pick shares
travels with that experiment's token, with any percentage masked. **The score never reads
text.** The random control (``llm: random``) uses the same flips but picks a random untried
condition, for free.

The reference implementation's planner, file for file, so a change there ports here as the
same change:

=================  ==========================================================================
``client.py``      the Responses API, an OpenAI-compatible chat server (vLLM), a mock; the
                   call cache; prices
``policy.py``      the coin, the planner, the random control, :func:`build_policy`
``prompts.py``     the system and user prompts
``schema.py``      the output schema, parsing, and the choice of the executed condition
``llm_suzuki.py``  the piece of the reference's ``src/llm_suzuki.py`` the above import
=================  ==========================================================================

The prompts and schema are the Suzuki study's, as in the reference. Changes from the reference
are marked ``# APPFL:``: the package-relative imports, a busy timeout on the shared cache, an
API key for a gateway behind ``LocalClient``, a configurable warmup (the reference fixes it at
5), and one planner per lab. Each APPFL agent owns its planner -- where the reference's single
process shared one -- so each lab writes its own call log, ``max_usd`` caps each lab rather
than the run, and an agent calls ``decide([context])`` with its own context only.

Status: implemented, not yet validated against the reference -- L3 is still changing there.

.. warning::
   The prompt carries this lab's **raw yields**: the model is asked to reason about them. Point
   ``llm_base_url`` at a model you host when the data is restricted.
"""

from .policy import L3Policy, RandomPolicy, build_policy

__all__ = ["L3Policy", "RandomPolicy", "build_policy"]
