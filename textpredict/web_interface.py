# textpredict/web_interface.py

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from textpredict import TextPredict

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    text: str
    task: str
    model: Optional[str] = None
    class_list: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    result: List


app = FastAPI()
tp = TextPredict()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Perform prediction on the input text.

    Args:
        request (PredictionRequest): The request containing text, task, model, and class_list.

    Returns:
        PredictionResponse: The prediction result.

    Raises:
        HTTPException: If there is an error during prediction.
    """
    try:
        logger.info(f"Received prediction request for task: {request.task}")
        result = tp.analyse(request.text, request.task, request.class_list)
        return PredictionResponse(result=result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
