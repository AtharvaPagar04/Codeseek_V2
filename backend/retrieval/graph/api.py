"""FastAPI route registration for repo graph endpoints."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Cookie, Header, HTTPException, Query

from retrieval.graph.store import (
    get_file_graph,
    get_graph_code_block,
    get_graph_overview,
    get_graph_tree,
    get_graph_node_details,
    get_latest_retrieval_trace,
    list_retrieval_trace_messages,
    get_retrieval_trace,
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

    @router.get("/sessions/{session_id}/graph")
    def session_graph(
        session_id: str,
        mode: str = Query(default="imports"),
        node_types: str | None = Query(default=None),
        edge_types: str | None = Query(default=None),
        search: str | None = Query(default=None),
        limit_nodes: int = Query(default=250, ge=1, le=500),
        limit_edges: int = Query(default=500, ge=1, le=1000),
        depth: int = Query(default=2, ge=1, le=3),
        focus_node_id: str | None = Query(default=None),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        node_types_list = [t.strip() for t in node_types.split(",") if t.strip()] if node_types else None
        edge_types_list = [t.strip() for t in edge_types.split(",") if t.strip()] if edge_types else None

        from retrieval.graph.visualization import get_session_graph_visualization
        data = get_session_graph_visualization(
            session_id=session_id,
            mode=mode,
            node_types=node_types_list,
            edge_types=edge_types_list,
            search=search,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
            depth=depth,
            focus_node_id=focus_node_id,
        )
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

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

    @router.get("/sessions/{session_id}/graph/node/{node_id}/details")
    def graph_node_details(
        session_id: str,
        node_id: str,
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        data = get_graph_node_details(session_id, node_id)
        if data["node"] is None:
            raise HTTPException(status_code=404, detail="Graph node not found")
        return data

    @router.get("/sessions/{session_id}/graph/code-blocks/{chunk_id}")
    def graph_code_block(
        session_id: str,
        chunk_id: str,
        max_lines: int = Query(default=250, ge=1, le=250),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        return get_graph_code_block(session_id, chunk_id, max_lines=max_lines)

    @router.get("/sessions/{session_id}/retrieval-trace/latest")
    def latest_retrieval_trace(
        session_id: str,
        thread_id: str | None = Query(default=None),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        return get_latest_retrieval_trace(session_id, thread_id=thread_id)

    @router.get("/sessions/{session_id}/retrieval-trace/messages")
    def retrieval_trace_messages(
        session_id: str,
        thread_id: str | None = Query(default=None),
        limit: int = Query(default=30, ge=1, le=100),
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        return list_retrieval_trace_messages(session_id, thread_id=thread_id, limit=limit)

    @router.get("/sessions/{session_id}/retrieval-trace/{assistant_message_id}")
    def retrieval_trace_for_message(
        session_id: str,
        assistant_message_id: str,
        session_token: str | None = Cookie(default=None, alias=auth_cookie_name),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _visible_session_or_404(session_id, session_token, authorization)
        data = get_retrieval_trace(session_id, assistant_message_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Retrieval trace not found")
        return data
