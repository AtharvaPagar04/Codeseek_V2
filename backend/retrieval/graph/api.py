"""FastAPI route registration for Phase 1 repo graph endpoints."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Cookie, Header, HTTPException, Query

from retrieval.graph.store import (
    get_file_graph,
    get_graph_overview,
    get_graph_tree,
    get_node_neighbors,
)


def register_graph_routes(
    router: APIRouter,
    *,
    require_auth_user: Callable,
    get_session: Callable[[str], dict | None],
    session_visible_to_user: Callable[[dict, dict | None], bool],
    auth_cookie_name: str,
) -> None:
    """Register graph endpoints on the provided API router."""

    def _visible_session_or_404(session_id: str, session_token: str | None, authorization: str | None) -> dict:
        auth_user = require_auth_user(session_token, authorization)
        session = get_session(session_id)
        if not session or not session_visible_to_user(session, auth_user):
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @router.get("/sessions/{session_id}/graph/tree")
    def graph_tree(
        session_id: str,
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        return get_graph_tree(session_id)

    @router.get("/sessions/{session_id}/graph/overview")
    def graph_overview(
        session_id: str,
        max_nodes: int = Query(default=400, ge=1, le=500),
        max_edges: int = Query(default=1000, ge=1, le=1200),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        return get_graph_overview(session_id, max_nodes=max_nodes, max_edges=max_edges)

    @router.get("/sessions/{session_id}/graph/file")
    def graph_file(
        session_id: str,
        path: str = Query(..., min_length=1),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        data = get_file_graph(session_id, path)
        if data["file"] is None:
            raise HTTPException(status_code=404, detail="Graph file node not found")
        return data

    @router.get("/sessions/{session_id}/graph/node/{node_id}/neighbors")
    def graph_node_neighbors(
        session_id: str,
        node_id: str,
        edge_type: str | None = Query(default=None),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        data = get_node_neighbors(session_id, node_id, edge_type=edge_type)
        if data["center"] is None:
            raise HTTPException(status_code=404, detail="Graph node not found")
        return data
