import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data_processing import (
    build_preprocessor,
    validate_data,
    prepare_input,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)
from src.predict import get_risk_label, LoanDefaultPredictor


# DADOS DE TESTE
@pytest.fixture
def sample_row() -> dict:
    return {
        "credit.policy": 1,
        "purpose": "debt_consolidation",
        "int.rate": 0.12,
        "installment": 400.0,
        "log.annual.inc": 11.0,
        "dti": 15.0,
        "fico": 700,
        "days.with.cr.line": 3600.0,
        "revol.bal": 12000.0,
        "revol.util": 40.0,
        "inq.last.6mths": 1,
        "delinq.2yrs": 0,
        "pub.rec": 0,
    }


@pytest.fixture
def sample_df(sample_row) -> pd.DataFrame:
    rows = [sample_row.copy() for i in range(20)]
    df = pd.DataFrame(rows)
    df[TARGET] = [0] * 16 + [1] * 4

    # RETORNA
    return df


# VALIDAÇÃO DE DADOS
def test_validate_data(sample_df):
    """VALIDAR SE OS DADOS ESTÃO DE ACORDO COM O ESPERADO"""
    validate_data(sample_df)


def test_validate_data_missing_column(sample_df):
    """VALIDAR SE TODAS AS COLUNAS OBRIGATÓRIAS ESTÃO PRESENTES"""
    df = sample_df.drop(columns=["fico"])

    with pytest.raises(ValueError, match="Colunas obrigatórias faltando"):
        validate_data(df)


# VALIDAÇÃO DO PRÉ-PROCESSADOR
def test_preprocessor_output_shape(sample_df):
    """VALIDAR A ESTRUTURA DE SAÍDA DO PRÉ-PROCESSADOR"""
    preprocessor = build_preprocessor()

    x = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_transformed = preprocessor.fit_transform(x)

    assert x_transformed.shape == (len(sample_df), 18)


def test_prepare_input_returns_dataframe(sample_row):
    """VALIDAR SE RETORNA UMA ÚNICA LINHA DO DATAFRAME"""
    df = prepare_input(sample_row)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


# RISCO X LABELS
@pytest.mark.parametrize(
    "prob, expected",
    [
        (0.05, "Risco Muito Baixo"),
        (0.25, "Risco Baixo"),
        (0.50, "Risco Moderado"),
        (0.70, "Risco Alto"),
        (0.90, "Risco Muito Alto"),
    ],
)
def test_risk_labels(prob, expected):
    """VALIDAR SE A CATEGORIZAÇÃO DE RISCO ESTÁ DE ACORDO COM O ESPERADO"""
    assert get_risk_label(prob) == expected


# PREDIÇÃO
def test_predict_return_valid_response(sample_row):
    """VALIDAR A ESTRUTURA DE DICIONÁRIO NO RETORNO DO MODELO"""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 18))

    with patch("src.predict.joblib.load", side_effect=[mock_model, mock_preprocessor]):
        predictor = LoanDefaultPredictor(
            "fake_model.joblib", "fake_pre.joblib", threshold=0.4
        )
        result = predictor.predict(sample_row)

    assert "prediction" in result
    assert "probability" in result
    assert "risk_label" in result
    assert "label" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
