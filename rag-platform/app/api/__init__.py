from .routes_docs import router as docs_router
from .routes_transcript import router as transcript_router
from .routes_query import router as query_router

__all__ = ["docs_router", "transcript_router", "query_router"]
