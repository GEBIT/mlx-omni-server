from time import time
from fastapi import APIRouter, HTTPException

from .embeddings_service import EmbeddingsService
from .schema import EmbeddingRequest, EmbeddingResponse
from mlx_omni_server.utils.gebit import check_model_allowed
from mlx_omni_server.utils.logger import logger

router = APIRouter(tags=["embeddings"])
embeddings_service = EmbeddingsService()


@router.post("/embeddings", response_model=EmbeddingResponse)
@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for text input.

    This endpoint generates vector representations of input text,
    which can be used for semantic search, clustering, and other NLP tasks.
    """
    check_model_allowed(request.model)
    logger.debug(f"Beginning embedding request for input lengths {[len(txt) for txt in (request.input if type(request.input) == list else [request.input])]}")
    start_time = time()
    try:
        result = embeddings_service.generate_embeddings(request)
        logger.debug(f"Completed embedding request after {time() - start_time:.3f}s")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
