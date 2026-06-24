from rag_ingestion.models.file import FileRecord
from rag_ingestion.stages.filtering import filter_files
from rag_ingestion.utils.counters import PipelineCounters


def make_file(relative_path: str) -> FileRecord:
    extension = ""
    if "." in relative_path.rsplit("/", 1)[-1]:
        extension = "." + relative_path.rsplit(".", 1)[-1]

    return FileRecord(
        path=f"/fake/repo/{relative_path}",
        relative_path=relative_path,
        extension=extension,
        size_bytes=100,
    )


def test_ignores_root_ingestion_state_file(tmp_path):
    files = [
        make_file(".rag_ingestion_state.json"),
        make_file("package.json"),
    ]
    counters = PipelineCounters()

    filtered = filter_files(files, str(tmp_path), counters)

    paths = {file.relative_path for file in filtered}

    assert ".rag_ingestion_state.json" not in paths
    assert "package.json" in paths
    assert counters.files_ignored == 1


def test_ignores_nested_ingestion_state_file(tmp_path):
    files = [
        make_file("backend/.rag_ingestion_state.json"),
        make_file("backend/package.json"),
    ]
    counters = PipelineCounters()

    filtered = filter_files(files, str(tmp_path), counters)

    paths = {file.relative_path for file in filtered}

    assert "backend/.rag_ingestion_state.json" not in paths
    assert "backend/package.json" in paths
    assert counters.files_ignored == 1


def test_keeps_normal_json_files(tmp_path):
    files = [
        make_file("package.json"),
        make_file("tsconfig.json"),
        make_file("backend/docs/example.json"),
    ]
    counters = PipelineCounters()

    filtered = filter_files(files, str(tmp_path), counters)

    paths = {file.relative_path for file in filtered}

    assert "package.json" in paths
    assert "tsconfig.json" in paths
    assert "backend/docs/example.json" in paths
    assert counters.files_ignored == 0
