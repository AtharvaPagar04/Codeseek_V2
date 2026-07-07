import asyncio
import inspect

from fastapi import APIRouter, FastAPI, HTTPException

from retrieval.graph.api import register_graph_routes
from retrieval.graph.builder import rebuild_session_hierarchy_graph
from retrieval.graph.ids import external_package_node_id, file_node_id


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


def test_graph_file_api_includes_imports_imported_by_and_unresolved(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    user_id = "graph-import-api-user"
    session_id = insert_session(user_id=user_id)
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=[
                "from utils.helpers import helper",
                "import requests",
                "from missing.module import X",
            ],
        ),
        make_chunk(chunk_id="helper-file", relative_path="utils/helpers.py", chunk_type="file"),
        make_chunk(
            chunk_id="helper-func",
            relative_path="utils/helpers.py",
            chunk_type="function",
            symbol_name="helper",
            qualified_symbol="utils/helpers.py::helper",
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    add_session_chunks(session_id, "utils/helpers.py", ["helper-file", "helper-func"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    app = _graph_test_app(session_id, user_id)

    file_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/file")
    app_file = _call(
        file_endpoint,
        session_id=session_id,
        path="app.py",
        session_token="test-token",
        authorization=None,
    )

    assert app_file["file"]["id"] == file_node_id(session_id, "app.py")
    assert any(item["target"]["id"] == external_package_node_id(session_id, "requests") for item in app_file["imports"])
    assert app_file["external_packages"][0]["name"] == "requests"
    assert app_file["unresolved_imports"][0]["raw_reference"] == "from missing.module import X"

    helper_file = _call(
        file_endpoint,
        session_id=session_id,
        path="utils/helpers.py",
        session_token="test-token",
        authorization=None,
    )
    assert any(item["source"]["id"] == file_node_id(session_id, "app.py") for item in helper_file["imported_by"])


def test_graph_neighbors_api_filters_import_edges(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    user_id = "graph-import-neighbors-user"
    session_id = insert_session(user_id=user_id)
    chunks = [
        make_chunk(
            chunk_id="app-file",
            relative_path="app.py",
            chunk_type="file",
            imports=["import requests"],
        ),
    ]
    add_session_chunks(session_id, "app.py", ["app-file"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    app = _graph_test_app(session_id, user_id)

    neighbors_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/node/{node_id}/neighbors")
    neighbor_data = _call(
        neighbors_endpoint,
        session_id=session_id,
        node_id=file_node_id(session_id, "app.py"),
        edge_type="imports",
        session_token="test-token",
        authorization=None,
    )

    assert neighbor_data["center"]["id"] == file_node_id(session_id, "app.py")
    assert neighbor_data["edges"]
    assert {edge["edge_type"] for edge in neighbor_data["edges"]} == {"imports"}


def test_graph_import_api_preserves_session_visibility_check(
    insert_session,
    add_session_chunks,
    make_chunk,
):
    user_id = "graph-import-auth-user"
    session_id = insert_session(user_id=user_id)
    chunks = [make_chunk(chunk_id="app-file", relative_path="app.py", chunk_type="file")]
    add_session_chunks(session_id, "app.py", ["app-file"])
    rebuild_session_hierarchy_graph(session_id, chunks)
    app = _graph_test_app(session_id, user_id)

    file_endpoint = _endpoint(app, "/api/v1/sessions/{session_id}/graph/file")
    try:
        _call(
            file_endpoint,
            session_id="missing-session",
            path="app.py",
            session_token="test-token",
            authorization=None,
        )
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("Expected session visibility failure")
