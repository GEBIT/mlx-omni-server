import os
from fastapi import HTTPException

def check_model_allowed(model_name: str) -> None:
    allowed_model = os.environ.get("ALLOWED_MODEL")
    if allowed_model is not None and model_name != allowed_model:
        raise HTTPException(status_code=400, detail=f"Requested model {model_name} but this worker only serves {allowed_model}")