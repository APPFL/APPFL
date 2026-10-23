"""ADKO-H unit tests. ``python -m unittest discover -s tests/decentralized -t .``

Offline and fast: the only language model used is the mock planner, and the molecule tests skip
themselves unless ``ADKO_PMO_POOL`` points at a pool. What is actually checked is the part a
GPU run cannot afford to get wrong -- that a campaign's verdict comes from the data, that the
claim channel deduplicates and expires, that a region built from a model's output is always
legal, and that a seed reproduces a run exactly.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from appfl.decentralized.algorithm.adko_h import metrics as hmetrics
from appfl.decentralized.algorithm.adko_h import rules as hrules
from appfl.decentralized.algorithm.adko_h.agent import AgentConfig
from appfl.decentralized.algorithm.adko_h.campaign import (
    Campaign,
    Evidence,
    Test,
    random_design,
)
from appfl.decentralized.algorithm.adko_h.claims import Claim, ClaimStore, Verdict
from appfl.decentralized.algorithm.adko_h.llm import build_planner
from appfl.decentralized.algorithm.adko_h.llm import schema as hschema
from appfl.decentralized.algorithm.adko_h.loop import RunConfig, run
from appfl.decentralized.algorithm.adko_h.problems.synthetic import Synthetic
from appfl.decentralized.algorithm.adko_h.region import Dim, Region, RegionSpace
from appfl.decentralized.algorithm.adko_h.surrogate import GP
from appfl.decentralized.algorithm.adko_h.token import HToken, encode
from appfl.decentralized.algorithm.adko_v2.knowledge_token import Quantizer
from appfl.decentralized.topology import build_topology


def space2() -> RegionSpace:
    return RegionSpace(
        [
            Dim("x", "real", 0.0, 10.0, meaning="first"),
            Dim("s", "cat", options=("a", "b", "c"), meaning="second"),
        ]
    )


class TestRegion(unittest.TestCase):
    def setUp(self):
        self.sp = space2()
        self.F = np.array([[1.0, 0.0], [5.0, 1.0], [9.0, 2.0], [5.0, 2.0]])

    def test_mask_and_full(self):
        r = self.sp.region(bounds={"x": (4.0, 6.0)}, picks={"s": [1, 2]})
        np.testing.assert_array_equal(r.mask(self.sp, self.F), [False, True, False, True])
        np.testing.assert_array_equal(self.sp.full().mask(self.sp, self.F), [True] * 4)

    def test_bad_input_is_clipped_not_raised(self):
        r = self.sp.region(
            bounds={"x": (-50.0, 500.0), "nope": (1, 2), "s": (0, 1)},
            picks={"s": [9, 1, -3], "x": [0]},
        )
        self.assertEqual(r.bounds, {"x": (0.0, 10.0)})
        self.assertEqual(r.picks, {"s": (1,)})

    def test_reversed_interval_is_sorted(self):
        self.assertEqual(self.sp.region(bounds={"x": (7.0, 2.0)}).bounds["x"], (2.0, 7.0))

    def test_empty_interval_dropped(self):
        self.assertEqual(self.sp.region(bounds={"x": (3.0, 3.0)}).bounds, {})

    def test_share_and_overlap(self):
        r = self.sp.region(bounds={"x": (0.0, 5.0)}, picks={"s": [0]})
        self.assertAlmostEqual(r.share(self.sp), 0.5 * (1 / 3))
        self.assertAlmostEqual(self.sp.full().share(self.sp), 1.0)
        self.assertAlmostEqual(r.overlap(self.sp, r), 1.0)
        far = self.sp.region(bounds={"x": (6.0, 10.0)}, picks={"s": [1]})
        self.assertAlmostEqual(r.overlap(self.sp, far), 0.0)

    def test_pack_round_trip(self):
        r = self.sp.region(bounds={"x": (2.0, 4.0)}, picks={"s": [0, 2]})
        back = self.sp.unpack(r.pack(self.sp))
        self.assertEqual(back.bounds, r.bounds)
        self.assertEqual(back.picks, r.picks)
        self.assertEqual(self.sp.unpack(self.sp.full().pack(self.sp)), Region())

    def test_label_is_human_readable(self):
        r = self.sp.region(bounds={"x": (2.0, 4.0)}, picks={"s": [0, 2]})
        self.assertEqual(r.label(self.sp), "x[2,4] s{a,c}")
        self.assertEqual(self.sp.full().label(self.sp), "whole space")

    def test_territory_boxes_cover_the_cube(self):
        centers = self.sp.centers(4, seed=0)
        self.assertEqual(centers.shape, (4, 2))
        masks = [self.sp.cell_mask(self.F, centers, i) for i in range(4)]
        np.testing.assert_array_equal(np.sum(masks, axis=0), [1, 1, 1, 1])


class TestCampaign(unittest.TestCase):
    def test_metrics(self):
        ev = Evidence(
            scores=np.array([0.1, 0.9]),
            descriptors=np.array([[0.0, 0.0], [3.0, 4.0]]),
            families=["a", "b", ""],
            known_descriptors=np.array([[0.0, 0.0]]),
            known_families={"a"},
            hull_before=1.0,
            hull_after=2.5,
        )
        self.assertEqual(Test("new_families").value(ev), 1.0)
        self.assertEqual(Test("novel_distance").value(ev), 5.0)
        self.assertAlmostEqual(Test("hull_growth").value(ev), 1.5)
        self.assertAlmostEqual(Test("max_objective").value(ev), 0.9)
        self.assertAlmostEqual(Test("mean_objective").value(ev), 0.5)

    def test_ops_and_validation(self):
        ev = Evidence(scores=np.array([0.3]))
        self.assertTrue(Test("max_objective", ">=", 0.2).passed(ev))
        self.assertTrue(Test("max_objective", "<=", 0.5).passed(ev))
        self.assertFalse(Test("max_objective", ">=", 0.5).passed(ev))
        with self.assertRaises(ValueError):
            Test("not_a_metric")
        with self.assertRaises(ValueError):
            Test("new_families", "==", 1)

    def test_novel_distance_with_nothing_known_is_infinite(self):
        ev = Evidence(descriptors=np.array([[1.0]]), known_descriptors=np.zeros((0, 0)))
        self.assertEqual(Test("novel_distance").value(ev), float("inf"))

    def test_judge_is_idempotent_and_records_infinity_as_none(self):
        sp = space2()
        c = Campaign(region=sp.full(), test=Test("novel_distance", ">=", 1.0), p_success=0.75)
        ev = Evidence(descriptors=np.array([[1.0]]), known_descriptors=np.zeros((0, 0)))
        v = c.judge(ev)
        self.assertTrue(v["success"])
        self.assertIsNone(v["value"])
        self.assertIs(c.judge(Evidence()), v)
        self.assertAlmostEqual(c.brier(), 0.0625)

    def test_budget_is_clamped_except_for_the_no_planner_arm(self):
        sp = space2()
        self.assertEqual(Campaign(region=sp.full(), budget=99).budget, 10)
        self.assertEqual(Campaign(region=sp.full(), budget=1).budget, 2)
        self.assertEqual(Campaign(region=sp.full(), budget=10**9, source="none").budget, 10**9)

    def test_unknown_rule_falls_back(self):
        self.assertEqual(Campaign(region=space2().full(), rule="telepathy").rule, "adko")

    def test_random_design_is_legal(self):
        sp = space2()
        rng = np.random.default_rng(0)
        for _ in range(20):
            d = random_design(sp, rng)
            c = Campaign(**{k: v for k, v in d.items() if k != "source"}, source=d["source"])
            self.assertIn(c.rule, hrules.RULES)
            self.assertGreaterEqual(c.budget, 2)
            self.assertLessEqual(c.budget, 10)
            self.assertLessEqual(c.region.share(sp), 1.0)


class TestClaims(unittest.TestCase):
    def setUp(self):
        self.sp = space2()
        self.store = ClaimStore(self.sp, source=0)
        self.region = self.sp.region(bounds={"x": (0.0, 5.0)})

    def claim(self, origin=1, cid="a1c0", ttl=2, expires=10):
        return Claim(
            origin=origin,
            claim_id=cid,
            region=self.region.pack(self.sp),
            rule="adko",
            hypothesis="h",
            round=0,
            expires=expires,
            ttl=ttl,
        )

    def test_dedupe_and_relay_decrements_ttl(self):
        self.store.ingest([self.claim()], [])
        out_c, _ = self.store.outbox()
        self.assertEqual([c.ttl for c in out_c], [1])
        self.store.ingest([self.claim()], [])  # the same claim again
        self.assertEqual(self.store.outbox(), ([], []))
        self.assertEqual(self.store.totals["claims_new"], 1)
        self.assertEqual(self.store.totals["claims_in"], 2)

    def test_last_hop_is_not_forwarded(self):
        self.store.ingest([self.claim(ttl=1)], [])
        self.assertEqual(self.store.outbox(), ([], []))

    def test_own_claims_are_ignored_on_the_way_in(self):
        self.store.ingest([self.claim(origin=0)], [])
        self.assertEqual(self.store.totals["claims_new"], 0)

    def test_penalty_is_an_indicator_and_expires(self):
        self.store.ingest([self.claim(expires=5), self.claim(cid="a1c1", expires=5)], [])
        F = np.array([[1.0, 0.0], [9.0, 0.0]])
        np.testing.assert_array_equal(self.store.penalty(F, now=3), [1.0, 0.0])
        np.testing.assert_array_equal(self.store.penalty(F, now=6), [0.0, 0.0])

    def test_verdicts_are_kept_newest_first_and_capped(self):
        store = ClaimStore(self.sp, source=0, keep=3)
        for i in range(5):
            store.ingest([], [Verdict(origin=2, claim_id=f"c{i}", metric="new_families", value=i, success=bool(i % 2))])
        got = [v.claim_id for v in store.recent_verdicts()]
        self.assertEqual(got, ["c4", "c3", "c2"])
        self.assertIn("HELD", store.digest())

    def test_digests_never_crash_when_empty(self):
        self.assertEqual(self.store.digest(), "(nothing yet)")
        self.assertEqual(self.store.claim_digest(0), "(none)")


class TestRules(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.0, 1.0, 2.0])
        self.sigma = np.array([1.0, 0.5, 0.25])
        self.G = np.array([0.0, 0.0, 1.0])
        self.L = np.array([1.0, 0.0, 0.0])
        self.nov = np.array([0.0, 5.0, 1.0])

    def test_each_rule_picks_what_it_says(self):
        v = lambda r: hrules.values(r, self.mu, self.sigma, self.G, self.L, self.nov)
        self.assertEqual(int(np.argmax(v("exploit"))), 2)
        self.assertEqual(int(np.argmax(v("explore"))), 0)
        self.assertEqual(int(np.argmax(v("novelty"))), 1)
        self.assertEqual(int(np.argmax(v("indep"))), 2)
        self.assertEqual(int(np.argmax(v("adko"))), 2)
        np.testing.assert_allclose(v("indep"), self.mu + 2.0 * self.sigma)

    def test_unknown_rule_is_the_settled_score(self):
        np.testing.assert_allclose(
            hrules.values("nonsense", self.mu, self.sigma, self.G, self.L),
            hrules.values("adko", self.mu, self.sigma, self.G, self.L),
        )

    def test_penalty_and_novelty_scale_with_sigma(self):
        base = hrules.values("exploit", self.mu, self.sigma, self.G, self.L)
        v, parts = hrules.adjust(base, self.sigma, penalty=np.array([1.0, 0, 0]), penalty_weight=2.0)
        u = float(np.sqrt(np.mean(self.sigma**2)))
        self.assertAlmostEqual(v[0], base[0] - 2.0 * u)
        self.assertAlmostEqual(parts["scale"], u)
        v2, _ = hrules.adjust(base, self.sigma, novelty=self.nov, novelty_weight=1.0)
        self.assertGreater(v2[1], base[1])

    def test_nearest_distance_handles_an_empty_history(self):
        d = hrules.nearest_distance(np.zeros((3, 2)), np.zeros((0, 2)))
        self.assertTrue(np.all(np.isinf(d)))
        d2 = hrules.nearest_distance(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0]]))
        np.testing.assert_allclose(d2, [0.0, 5.0])

    def test_pick_falls_back_when_the_region_is_empty(self):
        v = np.array([0.0, 9.0, 1.0])
        self.assertEqual(hrules.pick(v, np.array([True, False, True])), 2)
        self.assertEqual(hrules.pick(v, np.zeros(3, dtype=bool)), 1)


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.sp = space2()

    def test_format_mentions_every_coordinate(self):
        fmt = hschema.campaign_format(self.sp)
        props = fmt["schema"]["properties"]["region"]["properties"]
        self.assertEqual(set(props), {"x", "s"})
        self.assertEqual(set(fmt["schema"]["properties"]["region"]["required"]), {"x", "s"})

    def test_parse_a_good_answer(self):
        d = hschema.parse(
            self.sp,
            {
                "region": {"x": [2, 4], "s": None},
                "rule": "novelty",
                "budget": 6,
                "hypothesis": "h",
                "rationale": "r",
                "test": {"metric": "new_families", "op": ">=", "threshold": 2},
                "p_success": 0.6,
            },
        )
        self.assertEqual(d["region"].bounds, {"x": (2.0, 4.0)})
        self.assertEqual(d["region"].picks, {})
        self.assertEqual((d["rule"], d["budget"], d["source"]), ("novelty", 6, "llm"))
        self.assertEqual(d["test"].metric, "new_families")

    def test_parse_refuses_an_empty_answer(self):
        self.assertIsNone(hschema.parse(self.sp, {}))
        self.assertIsNone(hschema.parse(self.sp, None))
        self.assertIsNone(hschema.parse(self.sp, {"region": None, "rule": "adko"}))

    def test_parse_survives_nonsense_fields(self):
        d = hschema.parse(
            self.sp,
            {"region": {"x": "wide"}, "test": {"metric": "vibes", "op": "~", "threshold": "lots"}},
        )
        self.assertEqual(d["test"].metric, "new_families")
        self.assertEqual(d["test"].op, ">=")


class TestToken(unittest.TestCase):
    def test_round_trip_with_campaign_messages(self):
        tok = encode(
            agent_id="Agent1",
            source=1,
            round=3,
            x_shared=[0.5, 0.25],
            u=1.0,
            history=[0.0, 1.0],
            quantizer=Quantizer(8),
            new_best=True,
            insight="held",
            descriptor=[0.5, -0.25],
            family="scaffoldA",
            claims=[Claim(origin=1, claim_id="a1c0", region=[None, None])],
            verdicts=[Verdict(origin=1, claim_id="a1c0", success=True)],
        )
        back = HToken.deserialize(tok.serialize())
        self.assertEqual(back.family, "scaffoldA")
        self.assertEqual(back.descriptor, tok.descriptor)
        self.assertEqual(back.claims[0].claim_id, "a1c0")
        self.assertTrue(back.verdicts[0].success)
        self.assertTrue(back.is_best)
        self.assertLess(back.payload_bits(), back.size_bits())


class TestMetrics(unittest.TestCase):
    def test_cumulative_distinct_counts_budget_not_families(self):
        self.assertEqual(hmetrics.cumulative_distinct(["a", "a", "", "b"]), [1, 1, 1, 2])

    def test_n_circles(self):
        Z = np.array([[0.0], [0.5], [2.0], [4.0]])
        self.assertEqual(hmetrics.n_circles(Z, 1.0), 3)
        self.assertEqual(hmetrics.n_circles(np.zeros((0, 1)), 1.0), 0)

    def test_auc_top_k_rewards_finding_good_things_early(self):
        early = hmetrics.auc_top_k([1.0, 0.0, 0.0, 0.0], k=1)
        late = hmetrics.auc_top_k([0.0, 0.0, 0.0, 1.0], k=1)
        self.assertGreater(early, late)
        self.assertAlmostEqual(hmetrics.auc_top_k([0.5, 0.5], k=1), 0.5)

    def test_diverse_hits_ignores_repeats_of_one_family(self):
        self.assertEqual(hmetrics.diverse_hits([0.9] * 5, ["a"] * 5, 0.5), 1)
        self.assertEqual(hmetrics.diverse_hits([0.9, 0.1], ["a", "b"], 0.5), 1)

    def test_calibration_skill_against_the_base_rate(self):
        good = hmetrics.calibration([0.9, 0.1], [True, False])
        self.assertGreater(good["skill"], 0.0)
        flat = hmetrics.calibration([0.5, 0.5], [True, False])
        self.assertAlmostEqual(flat["skill"], 0.0)

    def test_hull_volume_degenerate(self):
        self.assertEqual(hmetrics.hull_volume(np.zeros((2, 3))), 0.0)


class TestTopology(unittest.TestCase):
    def test_random_regular_is_regular_and_connected(self):
        ids = [f"N{i}" for i in range(24)]
        t = build_topology("random_regular", ids, k=4, seed=1)
        self.assertTrue(all(len(t.neighbors(i)) == 4 for i in ids))
        d = t.describe()
        self.assertGreater(d["fiedler_value"], 1e-9)  # connected

    def test_odd_degree_times_odd_nodes_is_reduced(self):
        t = build_topology("random_regular", [f"N{i}" for i in range(5)], k=3, seed=0)
        self.assertEqual(t.k, 2)

    def test_same_seed_same_graph(self):
        ids = [f"N{i}" for i in range(12)]
        a = build_topology("random_regular", ids, k=4, seed=3)
        b = build_topology("random_regular", ids, k=4, seed=3)
        self.assertEqual([a.neighbors(i) for i in ids], [b.neighbors(i) for i in ids])


class TestRequestBody(unittest.TestCase):
    """What will actually hit vLLM. Checked here because a GPU hour is a bad place to find a
    malformed body, and nothing else in this suite talks to a server."""

    def setUp(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient

        sp = space2()
        self.policy = CampaignPolicy(MockClient(sp), sp, model="Qwen/Qwen3.8-27B-FP8")
        self.ctx = {
            "agent": 0,
            "round": 7,
            "problem": {"task": "t", "outcome": "o", "hints": "h"},
            "dims": sp.describe(),
            "rules": list(hrules.RULES),
            "budget_range": [2, 10],
            "metrics": ["new_families"],
            "n_evaluated": 3,
            "best": [{"point": "p", "score": 0.5, "family": "f"}],
            "my_families": ["f"],
            "peer_families": [],
            "territory": "x[0,5]",
            "first_campaign": True,
            "active_claims": "(none)",
            "verdicts": "(nothing yet)",
            "history": [],
        }

    def test_chat_body_is_what_vllm_expects(self):
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        body = chat_request(self.policy._request(self.ctx))
        self.assertEqual(body["model"], "Qwen/Qwen3.8-27B-FP8")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], "campaign")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertGreater(body["max_tokens"], 1000)
        self.assertIn("campaign", body["messages"][0]["content"].lower())
        self.assertIn("x[0,5]", body["messages"][1]["content"])

    def test_thinking_off_sends_no_thinking_keys(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        sp = space2()
        body = chat_request(
            CampaignPolicy(MockClient(sp), sp, thinking=None)._request(self.ctx)
        )
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_same_prompt_same_seed(self):
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        a = chat_request(self.policy._request(self.ctx))
        b = chat_request(self.policy._request(self.ctx))
        self.assertEqual(a["seed"], b["seed"])
        other = dict(self.ctx, round=8)
        self.assertNotEqual(chat_request(self.policy._request(other))["seed"], a["seed"])

    def test_a_fenced_or_chatty_reply_still_parses(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import _json

        self.assertEqual(_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_json('Here you go:\n{"a": 1}\nhope that helps'), {"a": 1})
        self.assertIsNone(_json("no json at all"))
        self.assertIsNone(_json(""))

    def test_the_mock_answer_satisfies_the_parser(self):
        import asyncio

        from appfl.decentralized.algorithm.adko_h.llm.policy import _json

        resp = asyncio.run(self.policy.client.complete(self.policy._request(self.ctx), self.ctx))
        design = hschema.parse(self.policy.space, _json(resp["text"]))
        self.assertIsNotNone(design)
        self.assertIn(design["rule"], hrules.RULES)


class _FakeRaw:
    def __init__(self, text, cost):
        from types import SimpleNamespace as NS

        self.headers = {"x-litellm-response-cost": str(cost)}
        msg = NS(content=text, model_extra={})
        self._resp = NS(
            id="fake",
            choices=[NS(message=msg, finish_reason="stop")],
            usage=NS(prompt_tokens=100, completion_tokens=50),
        )

    def parse(self):
        return self._resp


class _FakeAPI:
    """Stands in for the gateway: records request bodies, returns a canned design."""

    def __init__(self, text, cost=0.01):
        from types import SimpleNamespace as NS

        self.bodies = []
        outer = self

        async def create(**body):
            outer.bodies.append(body)
            return _FakeRaw(text, cost)

        self.chat = NS(completions=NS(with_raw_response=NS(create=create)))


class TestMag(unittest.TestCase):
    """The MAG path without the network: what is sent, what is charged, when it stops."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.sp = space2()
        self.design = (
            '{"region": {"x": [1, 3], "s": null}, "rule": "explore", "budget": 4, '
            '"hypothesis": "h", "rationale": "r", '
            '"test": {"metric": "new_families", "op": ">=", "threshold": 1}, "p_success": 0.4}'
        )

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, cap=None, cost=0.01):
        from appfl.decentralized.algorithm.adko_h.llm.mag import Ledger, MagClient, Slots

        ledger = Ledger(os.path.join(self.tmp.name, "ledger.txt"), cap)
        slots = Slots(n=2, gap=0.0, root=os.path.join(self.tmp.name, "slots"))
        return MagClient(slots=slots, ledger=ledger, api=_FakeAPI(self.design, cost))

    def ctx(self, agent=0):
        return {"agent": agent, "round": 3, "problem": {}, "dims": "", "budget_range": [2, 10],
                "metrics": [], "best": [], "history": []}

    def test_vllm_only_fields_are_not_sent_and_cost_is_recorded(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        c = self.client(cost=0.02)
        pol = CampaignPolicy(c, self.sp, model="claude-x@amazon", thinking="high", temperature=None)
        designs = pol.plan([self.ctx(0), self.ctx(1)])
        self.assertEqual([d["rule"] for d in designs], ["explore", "explore"])
        for body in c.api.bodies:
            for key in ("seed", "chat_template_kwargs", "reasoning_effort", "temperature"):
                self.assertNotIn(key, body)
            self.assertEqual(body["response_format"]["json_schema"]["name"], "campaign")
        self.assertAlmostEqual(pol.totals["cost_usd"], 0.04)
        self.assertAlmostEqual(c.ledger.total(), 0.04)

    def test_the_cap_stops_the_run_instead_of_falling_back(self):
        from appfl.decentralized.algorithm.adko_h.llm.mag import BudgetSpent
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        c = self.client(cap=0.015, cost=0.01)
        pol = CampaignPolicy(c, self.sp, model="m", temperature=None)
        pol.plan([self.ctx(0)])  # $0.01 spent, under the cap
        pol.plan([self.ctx(1)])  # $0.02 spent, now over
        with self.assertRaises(BudgetSpent):
            pol.plan([self.ctx(2)])

    def test_the_token_never_reaches_a_log(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        secret = "sk-not-a-real-token-but-a-canary"
        old = os.environ.get("AMSC_TOKEN")
        os.environ["AMSC_TOKEN"] = secret
        try:
            pol = CampaignPolicy(self.client(), self.sp, run_dir=os.path.join(self.tmp.name, "run"),
                                 model="m", temperature=None)
            pol.plan([self.ctx(0)])
            pol.close()
            for path in Path(self.tmp.name).rglob("*"):
                if path.is_file() and path.suffix in (".jsonl", ".txt", ".json"):
                    self.assertNotIn(secret, path.read_text())
        finally:
            if old is None:
                os.environ.pop("AMSC_TOKEN", None)
            else:
                os.environ["AMSC_TOKEN"] = old

    def test_one_event_loop_across_rounds(self):
        """An async HTTP client is bound to the loop it first ran on; a planner that opened a
        new loop per round broke every call after the first round against a real server."""
        import asyncio

        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        design = self.design

        class LoopBound:
            name = "loopbound"
            loop = None

            async def complete(self, request, ctx=None):
                here = asyncio.get_running_loop()
                if LoopBound.loop is None:
                    LoopBound.loop = here
                if here is not LoopBound.loop or LoopBound.loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                return {"text": design, "usage": {}, "status": "stop"}

        pol = CampaignPolicy(LoopBound(), self.sp, model="m")
        for rnd in range(4):
            pol.plan([self.ctx(a) for a in range(3)])
        self.assertEqual(pol.totals["errors"], 0)
        pol.close()

    def test_a_planner_that_keeps_failing_stops_the_run(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        class Broken:
            name = "broken"

            async def complete(self, request, ctx=None):
                raise ConnectionError("gateway unreachable")

        pol = CampaignPolicy(Broken(), self.sp, model="m")
        with self.assertRaises(RuntimeError):
            for i in range(4):
                pol.plan([self.ctx(a) for a in range(4)])


def short(arm: str, seed: int = 0, rounds: int = 8, **kw):
    """One tiny run of an arm, with the knobs the arms differ in."""
    problem = Synthetic(seed=seed)
    planner = build_planner(problem.region_space, kw.pop("planner_kind", "none"), seed=seed)
    cfg = RunConfig(
        rounds=rounds,
        agents=3,
        seed=seed,
        topology="random_regular",
        topology_kwargs={"k": 2, "seed": 0},
        planner=kw.pop("planner", "none"),
        label=arm,
        start_points=6,
        agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, **kw),
    )
    rec = run(problem, GP, cfg, planner=planner)
    if planner is not None:
        planner.close()
    return rec


class TestLoop(unittest.TestCase):
    def test_plain_adko_never_judges_a_campaign(self):
        rec = short("adko")
        self.assertEqual(rec["campaigns"]["n"], 0)
        self.assertEqual(rec["endpoints"]["evaluations"], 8 * 3)

    def test_arms_run_and_report_endpoints(self):
        for arm, kw in (
            ("adko", {}),
            ("adko-div", {"novelty_weight": 1.0}),
            ("random-campaign", {"planner": "random"}),
            ("adkoh", {"planner": "llm", "planner_kind": "mock"}),
        ):
            with self.subTest(arm=arm):
                rec = short(arm, **kw)
                e = rec["endpoints"]
                self.assertGreater(e["distinct_families"], 0)
                self.assertEqual(e["evaluations"], 24)
                self.assertEqual(e["redundancy"], 0)  # nothing evaluated twice

    def test_campaigns_are_judged_and_scored(self):
        rec = short("adkoh", rounds=14, planner="llm", planner_kind="mock")
        c = rec["campaigns"]
        self.assertGreater(c["n"], 0)
        self.assertIsNotNone(c["calibration"]["brier"])
        self.assertTrue(set(c["rules"]) <= set(hrules.RULES))

    def test_one_seed_reproduces_exactly(self):
        a = short("adkoh", planner="llm", planner_kind="mock")
        b = short("adkoh", planner="llm", planner_kind="mock")
        self.assertEqual(a["endpoints"], b["endpoints"])

    def test_different_seeds_differ(self):
        a = short("adko", seed=0)
        b = short("adko", seed=1)
        self.assertNotEqual(a["endpoints"]["hull_volume"], b["endpoints"]["hull_volume"])

    def test_claims_change_the_token_not_the_endpoints_shape(self):
        on = short("adkoh", planner="llm", planner_kind="mock", claims_on=True)
        off = short("adkoh", planner="llm", planner_kind="mock", claims_on=False)
        self.assertGreater(on["bits_per_token"], off["bits_per_token"])
        self.assertEqual(set(on["endpoints"]), set(off["endpoints"]))


class TestFootprint(unittest.TestCase):
    """The novelty rule has to see the whole team, not just the FIFO window."""

    def agent(self, memory_rounds=1):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        problem = Synthetic(seed=0)
        return ADKOHAgent(
            "Agent0",
            0,
            GP(),
            problem,
            AgentConfig(memory_rounds=memory_rounds),
            neighbor_indices=[1],
            degree=1,
        )

    def test_peer_inputs_outlive_the_token_memory(self):
        a = self.agent(memory_rounds=1)
        for rnd in range(6):
            a.observe([encode(
                agent_id="Agent1", source=1, round=rnd, x_shared=[float(rnd), 0.0, 0.0, 0.0, 0.0],
                u=0.5, history=[0.5], quantizer=Quantizer(8),
            )])
        self.assertLessEqual(len(a.memory), 2)          # the FIFO forgot most of them
        self.assertEqual(len(a._peer_X), 6)             # the footprint did not
        known = a._known_inputs()
        self.assertEqual(len(known), 6)
        # A candidate sitting on a long-forgotten peer point is not novel.
        d = hrules.nearest_distance(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]), known)
        self.assertAlmostEqual(float(d[0]), 0.0)


@unittest.skipUnless(os.environ.get("ADKO_PMO_POOL"), "set ADKO_PMO_POOL to run the PMO tests")
class TestPMO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from appfl.decentralized.algorithm.adko_h.problems.pmo import PMO

        cls.problem = PMO("qed", limit=2000)

    def test_pool_respects_a_region_and_excludes_what_is_run(self):
        sp = self.problem.region_space
        r = sp.region(bounds={"qed": (0.7, 1.0)})
        c = self.problem.pool(np.random.default_rng(0), 64, exclude=[], region=r)
        self.assertEqual(len(c), 64)
        self.assertTrue(r.mask(sp, c.F).all())
        again = self.problem.pool(np.random.default_rng(0), 64, exclude=list(c.key), region=r)
        self.assertEqual(len(set(again.key) & set(c.key)), 0)

    def test_evaluate_scores_and_names_a_scaffold(self):
        from appfl.decentralized.algorithm.adko_h.problem import Pick

        c = self.problem.pool(np.random.default_rng(1), 4)
        outs = self.problem.evaluate(
            [Pick(agent=0, key=int(k), X=c.X[i], F=c.F[i], meta=c.meta[i]) for i, k in enumerate(c.key)]
        )
        self.assertEqual(len(outs), 4)
        self.assertTrue(all(0.0 <= o.score <= 1.0 for o in outs))
        self.assertEqual(self.problem.descriptor_dim(), 3)
        self.assertTrue(any(o.family for o in outs))

    def test_oracle_calls_are_cached(self):
        from appfl.decentralized.algorithm.adko_h.problem import Pick

        c = self.problem.pool(np.random.default_rng(2), 2)
        picks = [Pick(agent=0, key=int(k), X=c.X[i], F=c.F[i], meta=c.meta[i]) for i, k in enumerate(c.key)]
        before = self.problem.calls
        self.problem.evaluate(picks)
        self.problem.evaluate(picks)
        self.assertEqual(self.problem.calls - before, 2)


if __name__ == "__main__":
    unittest.main()
