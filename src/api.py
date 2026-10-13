import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from src.predict import load_model, predict_with_model
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
    app.state.model = None


app = FastAPI(
    title="Support Ticket Classification API",
    description="Classifies IT support tickets into predefined categories.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None

    return {
        "status": "ok" if model_loaded else "unavailable",
        "model_loaded": model_loaded,
    }

@app.get("/health/live")
def liveness_check():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness_check(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None

    if not model_loaded:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    return {
        "status": "ready",
        "model_loaded": True,
    }


class TicketRequest(BaseModel):
    ticket_text: str = Field(
        min_length=1,
        description="The support ticket text to classify.",
        examples=["My laptop screen is not working"],
    )

    @field_validator("ticket_text")
    @classmethod
    def validate_ticket_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text cannot be empty or whitespace.")

        return value


class PredictionResponse(BaseModel):
    predicted_category: str
    probabilities: dict[str, float]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TicketRequest, http_request: Request):
    model = getattr(http_request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not ready.",
        )

    try:
        predicted_class, ranked_predictions = predict_with_model(
            model,
            request.ticket_text,
        )

        return PredictionResponse(
            predicted_category=predicted_class,
            probabilities={
                category: float(probability)
                for category, probability in ranked_predictions
            },
        )

    except Exception:
        logger.exception("Unexpected error during ticket prediction.")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the ticket.",
        ) from None