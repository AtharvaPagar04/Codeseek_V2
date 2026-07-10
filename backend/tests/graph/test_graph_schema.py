from retrieval.db import db_cursor, get_db_path
from retrieval.graph.ids import file_node_id, graph_edge_id, repo_node_id


def test_graph_db_fixture_uses_isolated_sqlite_path(graph_db):
    assert get_db_path() == graph_db.resolve()


def test_graph_schema_tables_and_indexes_exist(graph_db):
    with db_cursor() as (_conn, cursor):
        tables = {
            row["name"]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row["name"]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert "code_graph_nodes" in tables
    assert "code_graph_edges" in tables
    assert "code_graph_builds" in tables
    assert "idx_code_graph_nodes_session_type" in indexes
    assert "idx_code_graph_nodes_session_path" in indexes
    assert "idx_code_graph_nodes_session_qualified" in indexes
    assert "idx_code_graph_nodes_session_chunk" in indexes
    assert "idx_code_graph_edges_session_type" in indexes
    assert "idx_code_graph_edges_session_source" in indexes
    assert "idx_code_graph_edges_session_target" in indexes
    assert "idx_code_graph_edges_session_source_path" in indexes


def test_sqlite_foreign_keys_enabled_and_node_edge_cascade_works(graph_db):
    session_id = "session-1"
    source_id = repo_node_id(session_id)
    target_id = file_node_id(session_id, "src/app.py")
    edge_id = graph_edge_id(session_id, source_id, "contains", target_node_id=target_id)

    with db_cursor() as (conn, cursor):
        pragma = cursor.execute("PRAGMA foreign_keys").fetchone()[0]
        assert pragma == 1
        cursor.execute(
            """
            INSERT INTO code_graph_nodes (
                id, session_id, node_type, name, qualified_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, session_id, "repo", "repo", "repo", "now", "now"),
        )
        cursor.execute(
            """
            INSERT INTO code_graph_nodes (
                id, session_id, node_type, name, qualified_name, relative_path,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (target_id, session_id, "file", "app.py", "src/app.py", "src/app.py", "now", "now"),
        )
        cursor.execute(
            """
            INSERT INTO code_graph_edges (
                id, session_id, source_node_id, target_node_id, edge_type,
                confidence_tier, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (edge_id, session_id, source_id, target_id, "contains", "structural", "now"),
        )
        cursor.execute("DELETE FROM code_graph_nodes WHERE id = ?", (source_id,))
        edge_count = cursor.execute(
            "SELECT COUNT(*) AS c FROM code_graph_edges WHERE id = ?",
            (edge_id,),
        ).fetchone()["c"]

    assert edge_count == 0
