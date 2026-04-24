import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)

RISK_LABELS = {
    (0.0, 0.20): "Risco Muito Baixo",
    (0.20, 0.40): "Risco Baixo",
    (0.40, 0.60): "Risco Moderado",
    (0.60, 0.80): "Risco Alto",
    (0.80, 1.01): "Risco Muito Alto",
}


def get_risk_label(probability: float) -> str:
    """OBTÉM O RÓTULO DE RISCO COM BASE NA PROBABILIDADE"""
    for (low, high), label in RISK_LABELS.items():
        if low <= probability < high:
            return label

    return "Risco Muito Alto"


class LoanDefaultPredictor:
    """ENCAPSULA O PRÉ-PROCESSADOR E O MODELO TREINADO PARA INFERÊNCIA"""

    def __init__(self, model_path: str, preprocessor_path: str, threshold: float = 0.5):
        logger.info(f"Carregando modelo de: {model_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.threshold = threshold

        logger.info("Modelo Pronto!")

    def predict(self, dados_entrada: dict) -> dict:
        """EXECUTAR PREDIÇÃO"""
        df = pd.DataFrame([dados_entrada])
        x = self.preprocessor.transform(df)

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(x)[0, 1])
            prediction = int(proba >= self.threshold)

        else:
            prediction = int(self.model.predict(x)[0])
            proba = float(prediction)

        # RETORNA
        return {
            "prediction": prediction,
            "probability": round(proba, 4),
            "risk_label": get_risk_label(proba),
            "label": "Default" if prediction == 1 else "Fully Paid",
        }
