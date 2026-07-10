import os
from dataclasses import dataclass
from qdrant_client import QdrantClient

@dataclass(frozen=True)
class QdrantSettings:
    url: str | None
    api_key: str | None
    host: str
    port: int
    prefer_grpc: bool = False
    timeout: float | None = None

    @property
    def is_cloud(self) -> bool:
        return bool(self.url)

def get_qdrant_settings() -> QdrantSettings:
    url = (os.getenv("QDRANT_URL") or "").strip() or None
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
    host = (os.getenv("QDRANT_HOST") or "localhost").strip()
    port = int((os.getenv("QDRANT_PORT") or "6333").strip())
    timeout_raw = (os.getenv("QDRANT_TIMEOUT_SECONDS") or "").strip()
    timeout = float(timeout_raw) if timeout_raw else None
    
    return QdrantSettings(
        url=url,
        api_key=api_key,
        host=host,
        port=port,
        timeout=timeout,
    )

def create_qdrant_client(**extra_kwargs) -> QdrantClient:
    """Create a QdrantClient configured for either cloud or local operation.
    
    Any extra_kwargs will override the environment settings.
    """
    settings = get_qdrant_settings()
    
    if settings.url:
        kwargs = {"url": settings.url}
        if settings.api_key:
            kwargs["api_key"] = settings.api_key
        if settings.timeout is not None:
            kwargs["timeout"] = settings.timeout
        kwargs.update(extra_kwargs)
        return QdrantClient(**kwargs)

    kwargs = {"host": settings.host, "port": settings.port}
    if settings.timeout is not None:
        kwargs["timeout"] = settings.timeout
    kwargs.update(extra_kwargs)
    return QdrantClient(**kwargs)
