from retrieval.stores import chat_store
from retrieval import session_indexer


def test_append_list_and_clear_session_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("CODESEEK_DB_PATH", str(tmp_path / "codeseek.sqlite3"))
    monkeypatch.setattr(session_indexer, "WORKSPACE_ROOT", tmp_path / "repos")
    monkeypatch.setattr(session_indexer, "_enqueue_index_job", lambda _session_id: None)

    session = session_indexer.create_session("octocat/hello-world", "local")

    first = chat_store.append_message(session["id"], "user", "What is this project about?")
    second = chat_store.append_message(
        session["id"],
        "assistant",
        "This is a sample project.",
        sources=[{"relative_path": "README.md", "start_line": 1, "end_line": 10, "symbol_name": "README"}],
        context_tokens=123,
    )

    messages = chat_store.list_session_messages(session["id"])
    assert [message["id"] for message in messages] == [first["id"], second["id"]]
    assert messages[1]["sources"][0]["relative_path"] == "README.md"
    assert messages[1]["context_tokens"] == 123

    cleared = chat_store.clear_session_messages(session["id"])
    assert cleared == 2
    assert chat_store.list_session_messages(session["id"]) == []
