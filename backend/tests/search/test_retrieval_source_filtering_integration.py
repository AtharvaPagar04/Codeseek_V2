"""Integration-style checks for final source selection behavior."""

import unittest

from retrieval.search.source_filter import select_sources_for_display


class SourceFilteringIntegrationTests(unittest.TestCase):
    def test_caps_and_primary_priority(self) -> None:
        query = "Trace authenticated_get and sign_query request path"

        primary_sources = [
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "authenticated_get",
                "start_line": 210,
                "end_line": 248,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "sign_query",
                "start_line": 148,
                "end_line": 168,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "signed_params",
                "start_line": 170,
                "end_line": 189,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "auth_headers",
                "start_line": 191,
                "end_line": 198,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "account_info",
                "start_line": 250,
                "end_line": 260,
                "expansion_type": "primary",
            },
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "create_listen_key",
                "start_line": 284,
                "end_line": 301,
                "expansion_type": "primary",
            },
        ]

        expanded_sources = [
            {
                "relative_path": "backend/src/exchange/binance_rest_client.py",
                "symbol_name": "BinanceRestClient",
                "start_line": 14,
                "end_line": 301,
                "expansion_type": "parent_class",
            },
            {
                "relative_path": "backend/src/runtime/async_event_bus.py",
                "symbol_name": "AsyncEventBus",
                "start_line": 5,
                "end_line": 46,
                "expansion_type": "callee",
            },
            {
                "relative_path": "backend/src/core/runtime_builder.py",
                "symbol_name": "build_runtime_state",
                "start_line": 26,
                "end_line": 81,
                "expansion_type": "callee",
            },
        ]

        selected = select_sources_for_display(query, primary_sources + expanded_sources)

        # cap: max 5 primary + max 2 expanded
        self.assertLessEqual(len(selected), 7)

        # primaries should come first in display ordering
        seen_expanded = False
        for src in selected:
            if src["expansion_type"] != "primary":
                seen_expanded = True
            if seen_expanded:
                self.assertNotEqual(src["expansion_type"], "primary")

        primary_count = sum(1 for s in selected if s["expansion_type"] == "primary")
        expanded_count = sum(1 for s in selected if s["expansion_type"] != "primary")
        self.assertLessEqual(primary_count, 5)
        self.assertLessEqual(expanded_count, 2)


if __name__ == "__main__":
    unittest.main()
