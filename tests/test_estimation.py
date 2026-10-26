import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from insuranceai import price_search
from insuranceai.agent import InsuranceAgent
from insuranceai.estimation import (
    build_estimates,
    compute_breakdown,
    default_breakdown_dict,
    parse_plan,
    render_estimate_markdown,
)
from insuranceai.claims_calc import PolicyInput
from insuranceai.price_search import (
    PriceSearchSession,
    extract_prices,
    summarize_prices,
)


class FakeProvider:
    name = "fake"

    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self.error:
            raise self.error
        return self.rows


ROWS = [
    ("Bumper", "https://a.example/1", "Front bumper price ₹8,000 - ₹12,000 for hatchback"),
    ("Bumper 2", "https://b.example/2", "Genuine bumper Rs. 9,500 and aftermarket INR 6,500"),
    ("Junk", "https://c.example/3", "Unrelated ₹50 and ₹99,99,999 listings"),
]


class PriceParsingTests(unittest.TestCase):
    def test_extract_prices_handles_ranges_formats_and_bounds(self):
        prices = extract_prices("₹5,000 - ₹7,000, Rs. 1,200 and INR 40 plus ₹9,99,99,999", "parts")
        self.assertEqual(sorted(prices), [1200.0, 5000.0, 7000.0])
        self.assertEqual(extract_prices("no prices here", "parts"), [])

    def test_summarize_drops_outliers_and_widens_sparse_data(self):
        low, high, confidence = summarize_prices([5000, 6000, 7000, 8000, 9000, 400000])
        self.assertEqual((low, high, confidence), (5000, 9000, "medium"))
        low, high, confidence = summarize_prices([10000])
        self.assertEqual((low, high, confidence), (7500, 12500, "low"))
        self.assertIsNone(summarize_prices([]))


class PriceSessionTests(unittest.TestCase):
    def setUp(self):
        price_search._CACHE.clear()

    def test_search_returns_range_with_sources_and_uses_cache(self):
        provider = FakeProvider(ROWS)
        session = PriceSearchSession(provider, max_calls=5)
        first = session.search("front bumper", "parts", "replace", "Maruti Swift")
        second = session.search("front bumper", "parts", "replace", "Maruti Swift")
        self.assertEqual(first.status, "ok")
        self.assertLess(first.low, first.high)
        self.assertEqual(len(first.sources), 2)
        self.assertEqual(second.status, "ok")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(session.calls, 1)

    def test_call_cap_failure_and_unconfigured_never_raise(self):
        capped = PriceSearchSession(FakeProvider(ROWS), max_calls=1)
        capped.search("bumper", "parts", "replace", "car")
        self.assertEqual(capped.search("door", "parts", "replace", "car").status, "limit_reached")

        price_search._CACHE.clear()
        failing = PriceSearchSession(FakeProvider(error=RuntimeError("boom")), max_calls=3)
        self.assertEqual(failing.search("bumper", "parts", "replace", "car").status, "error")

        price_search._CACHE.clear()
        empty = PriceSearchSession(FakeProvider([("t", "u", "nothing")]), max_calls=3)
        self.assertEqual(empty.search("bumper", "parts", "replace", "car").status, "not_estimable")

        with patch.object(price_search, "get_provider", return_value=None):
            none = PriceSearchSession()
        self.assertFalse(none.configured)
        self.assertEqual(none.search("bumper", "parts", "replace", "car").status, "not_configured")

    def test_provider_selection_uses_existing_env_names(self):
        env = {"TAVILY_SEARCH_API": "t", "SERPER_SEARCH_API": "s", "PRICE_SEARCH_PROVIDER": "serper"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(price_search.get_provider().name, "serper")
        with patch.dict("os.environ", {"TAVILY_SEARCH_API": "t"}, clear=True):
            self.assertEqual(price_search.get_provider().name, "tavily")
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(price_search.get_provider())


def seeded_session():
    price_search._CACHE.clear()
    session = PriceSearchSession(FakeProvider(ROWS), max_calls=10)
    for component, part in (("parts", "front bumper"), ("labor", "front bumper"), ("parts", "radiator")):
        session.search(part, component, "replace", "Swift")
    return session


class EstimationTests(unittest.TestCase):
    def plan(self):
        return parse_plan(
            "```json\n"
            + json.dumps(
                {
                    "vehicle": "Maruti Swift",
                    "visible_damage": [
                        {"part": "front bumper", "material": "rubber_plastic", "operation": "replace",
                         "parts_query_id": "q1", "labor_query_id": "q2"},
                        {"part": "hood", "material": "metal", "operation": "replace",
                         "parts_query_id": "q99", "labor_query_id": None},
                    ],
                    "internal_damage": [
                        {"part": "radiator", "material": "metal", "operation": "replace",
                         "likelihood": "medium", "reason": "Frontal impact", "parts_query_id": "q3"}
                    ],
                }
            )
            + "\n```"
        )

    def test_numbers_only_from_recorded_searches_and_internal_kept_separate(self):
        cost, internal = build_estimates(self.plan(), seeded_session())
        bumper, hood = cost["items"]
        self.assertTrue(bumper["estimable"])
        self.assertLess(bumper["parts"]["low"], bumper["parts"]["high"])
        self.assertFalse(hood["estimable"])  # hallucinated query id is rejected
        self.assertEqual(cost["total"], bumper["subtotal"])
        self.assertFalse(internal["included_in_claimable"])
        self.assertEqual(internal["items"][0]["likelihood"], "medium")
        self.assertIsNotNone(internal["total"])

    def test_breakdown_excludes_internal_damage_and_recomputes_by_policy(self):
        cost, internal = build_estimates(self.plan(), seeded_session())
        default = default_breakdown_dict(cost)
        self.assertLessEqual(default["claimable"]["high"], cost["total"]["high"])
        zero_dep = compute_breakdown(cost, PolicyInput(template="comprehensive_zero_dep"))
        standard = compute_breakdown(cost, PolicyInput())
        self.assertGreaterEqual(zero_dep.claimable.high, standard.claimable.high)
        markdown = render_estimate_markdown(cost, internal, default)
        self.assertIn("Possible Internal Damage (Not Confirmed)", markdown)
        self.assertIn("Indicative estimate only", markdown)

    def test_parse_plan_tolerates_surrounding_text_and_extra_objects(self):
        plan = parse_plan(
            'Here is the plan: {"vehicle": "Swift", "visible_damage": [{"part": "hood"}], '
            '"internal_damage": []} Hope that helps {"note": "ignore me"}'
        )
        self.assertEqual(plan.visible_damage[0].part, "hood")
        from insuranceai.utils import _response_text

        blocks = [{"type": "text", "text": '{"visible_damage": [], "internal_damage": []}'}]
        self.assertEqual(parse_plan(_response_text(blocks)).visible_damage, [])

    def test_search_progress_hook_fires_and_failures_are_ignored(self):
        price_search._CACHE.clear()
        seen = []

        def hook(session, result):
            seen.append((result.part, len(session.results)))
            raise RuntimeError("hook failure must not break the search")

        session = PriceSearchSession(FakeProvider(ROWS), max_calls=4, on_result=hook)
        self.assertEqual(session.search("front bumper", "parts", "replace", "car").status, "ok")
        self.assertEqual(seen, [("front bumper", 1)])

    def test_parse_plan_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_plan("sorry, no json")
        with self.assertRaises(ValueError):
            parse_plan('{"visible_damage": [{"part": "  "}]}')


class StubModel:
    """Scripted chat model: one tool call, then the final JSON plan."""

    def __init__(self):
        self.turn = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_repair_prices", "id": "c1",
                             "args": {"part": "front bumper", "component": "parts",
                                      "operation": "replace", "vehicle": "Swift"}}],
            )
        return AIMessage(
            content=json.dumps(
                {"vehicle": "Swift", "visible_damage": [
                    {"part": "front bumper", "material": "rubber_plastic", "operation": "replace",
                     "parts_query_id": "q1"}], "internal_damage": []}
            )
        )


class AgentLoopTests(unittest.TestCase):
    def test_agent_tool_loop_records_results_in_session(self):
        from insuranceai import workflow
        from insuranceai.tools import get_pricing_tools

        price_search._CACHE.clear()
        session = PriceSearchSession(FakeProvider(ROWS), max_calls=5)
        agent = InsuranceAgent(StubModel(), get_pricing_tools(), system_prompt="x", log_tools=False)
        with patch.object(workflow, "PriceSearchSession", return_value=session), \
             patch.object(workflow, "_get_estimation_agent", return_value=agent), \
             patch.object(workflow, "encode_image_for_llm", return_value="data:image/png;base64,AA=="):
            cost, internal, breakdown, _ = workflow._run_estimation(
                {"predictions": [{"label": "dent", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]},
                "## Summary\nDent.",
                "unused.jpg",
            )
        self.assertEqual(cost["status"], "estimated")
        self.assertTrue(cost["items"][0]["estimable"])
        self.assertIsNotNone(breakdown)
        self.assertEqual(session.calls, 1)

    def test_estimation_degrades_gracefully_when_unconfigured_or_failing(self):
        from insuranceai import workflow

        unconfigured = PriceSearchSession(None, max_calls=1)
        with patch.object(price_search, "get_provider", return_value=None):
            unconfigured = PriceSearchSession()
        with patch.object(workflow, "PriceSearchSession", return_value=unconfigured):
            cost, internal, breakdown, tokens = workflow._run_estimation({}, "r", "x.jpg")
        self.assertEqual(cost["status"], "unavailable")
        self.assertIsNone(internal)

        class Broken:
            def invoke(self, *a, **k):
                raise RuntimeError("llm down")

        session = PriceSearchSession(FakeProvider(ROWS), max_calls=1)
        with patch.object(workflow, "PriceSearchSession", return_value=session), \
             patch.object(workflow, "_get_estimation_agent", return_value=Broken()), \
             patch.object(workflow, "encode_image_for_llm", return_value="data:image/png;base64,AA=="):
            cost, *_ = workflow._run_estimation({}, "r", "x.jpg")
        self.assertEqual(cost["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
