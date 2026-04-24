import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import LoanInput, PredictionResponse, HealthResponse
from src.predict import LoanDefaultPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
THRESHOLD = 0.4

# INSTANCIAR PREDICTOR
predictor: LoanDefaultPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """CARREGAR MODELO"""
    global predictor

    try:
        predictor = LoanDefaultPredictor(MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD)
        logger.info("Modelo carregado com sucesso ao iniciar a API")

    except FileNotFoundError as e:
        logger.error(
            f"Arquivo de modelo não localizado: {e}. Execute `python src/train.py` primeiro."
        )

    yield
    predictor = None
    logger.info("Aplicação encerrada")


# APP FASTAPI
app = FastAPI(
    title="API - Previsor de Inadimplência de Empréstimos",
    version="1.0.0",
    lifespan=lifespan,
)

# MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# HEALTH CHECK
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok" if predictor is not None else "model_not_loaded",
        model_loaded=predictor is not None,
    )


# PREDICT INDIVIDUAL
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(loan: LoanInput):
    """PREDIÇÃO DE UM ÚNICO CASO"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute `python src/train.py` e reinicie a API",
        )

    result = predictor.predict(loan.to_raw_dict())
    return PredictionResponse(**result)


# PREDICT EM LOTE
@app.post(
    "/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"]
)
def predict_batch(loans: list[LoanInput]):
    """PREDIÇÃO EM LOTE NO MÁXIMO 100"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    if len(loans) > 100:
        raise HTTPException(status_code=400, detail="O tamanho máximo do lote é 100")

    return [
        PredictionResponse(**predictor.predict(loan.to_raw_dict())) for loan in loans
    ]
