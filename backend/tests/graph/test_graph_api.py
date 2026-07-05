import asyncio
import inspect

from fastapi import APIRouter, FastAPI

from retrieval.graph.api import register_graph_routes
from retrieval.graph.builder import rebuild_session_hierarchy_graph
from retrieval.graph.ids import file_node_id, symbol_node_id


def _graph_test_app(session_id: str, user_id: str) -> FastAPI:
    app = FastAPI()
    router = APIRouter(prefix="/api/v1")

    def require_auth_user(session_token: str | None, authorization: str | None = None) -> dict:
        assert session_token == "test-token"
        return {"id": user_id}

    def get_session(current_session_id: str) -> dict | None:
        if current_session_id != session_id:
            return None
        return {"id": session_id, "user_id": user_id, "status": "ready"}

    def session_visible_to_user(session: dict, auth_user: dict | None) -> bool:
        return bool(auth_user and session.get("user_id") == auth_user["id"])

    register_graph_routes(
        router,
        require_auth_user=require_auth_user,
        get_session=get_session,
        session_visible_to_user=session_visible_to_user,
        auth_cookie_name="codeseek_session",
    )
    app.include_router(router)
    return app


def _endpoint(app: FastAPI, path: str):
    for route in app.routes:
        if getattr(route, "path", "") == path:
            return route.endpoint
    raise AssertionError(f"Route not registered: {path}")


def _call(endpoint, **kwargs):
    result = endpoint(**kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def test_graph_api_tree_overview_file_and_neighbors(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    user_id = "graph-api-user"
    session_id = insert_session(user_id=user_id)
    chunks = [
        make_chunk(chunk_id="file", relative_path="src/app.py", chunk_type="file"),
        make_chunk(
            chunk_id="func",
            relative_path="src/app.py",
            chunk_type="function",
            symbol_name="load",
            qualified_symbol="src/app.py::load",
        ),
    ]
    add_session_chunks(session_id, "src/app.py", ["file", "func"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    app = _graph_test_app(session_id, user_id)

    tree_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/tree")
    tree_data = _call(tree_endpoint, session_id=session_id, session_token="test-token", authorization=None)
    assert tree_data["root"]["node_type"] == "repo"

    overview_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/overview")
    overview_data = _call(
        overview_endpoint,
        session_id=session_id,
        max_nodes=400,
        max_edges=1000,
        session_token="test-token",
        authorization=None,
    )
    assert overview_data["stats"]["node_count"] >= 3
    assert overview_data["stats"]["edge_count"] >= 2
    assert overview_data["stats"]["truncated"] is False

    file_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/file")
    file_data = _call(
        file_endpoint,
        session_id=session_id,
        path="src/app.py",
        session_token="test-token",
        authorization=None,
    )
    assert file_data["file"]["id"] == file_node_id(session_id, "src/app.py")
    assert any(symbol["name"] == "load" for symbol in file_data["symbols"])

    node_id = symbol_node_id(session_id, "src/app.py", "src/app.py::load", "function")
    neighbors_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/node/{node_id}/neighbors")
    neighbor_data = _call(
        neighbors_endpoint,
        session_id=session_id,
        node_id=node_id,
        edge_type="defines",
        session_token="test-token",
        authorization=None,
    )
    assert neighbor_data["center"]["id"] == node_id
    assert any(edge["edge_type"] == "defines" for edge in neighbor_data["edges"])
