"""Smoke test for the local Qdrant ingestion output."""

from retrieval.support.qdrant_config import create_qdrant_client

from rag_ingestion.config import COLLECTION_NAME, EMBEDDING_DIM


def main() -> None:
    client = create_qdrant_client()

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points: {info.points_count}")

    if hasattr(client, "search"):
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=[0.0] * EMBEDDING_DIM,
            limit=3,
        )
    else:
        query = client.query_points(
            collection_name=COLLECTION_NAME,
            query=[0.0] * EMBEDDING_DIM,
            limit=3,
        )
        results = query.points

    for result in results:
        payload = result.payload or {}
        print(
            payload.get("symbol_name", ""),
            payload.get("relative_path", ""),
        )


if __name__ == "__main__":
    main()
