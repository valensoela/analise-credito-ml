import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "credit.policy",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]

CATEGORICAL_FEATURES = ["purpose"]

TARGET = "not.fully.paid"

PURPOSE_CATEGORIES = [
    "all_other",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "major_purchase",
    "small_business",
]


def load_data(path: str) -> pd.DataFrame:
    """CARREGAR DADOS CSV"""
    logger.info(f"Carregando dados de: {path}")
    df = pd.read_csv(path)

    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # RETORNA
    return df


def validate_data(df: pd.DataFrame) -> None:
    """VALIDAR BÁSICA DOS DADOS"""
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    columns_missing = [c for c in required_columns if c not in df.columns]

    # VERIFICA COLUNAS FALTANDO
    if columns_missing:
        raise ValueError(f"Colunas obrigatórias faltando: {columns_missing}")

    # VERIFICA CONTAGEM DE NULOS
    null_counts = df[required_columns].isnull().sum()

    if null_counts.any():
        logger.warning(f"Valores nulos localizados: \n{null_counts[null_counts > 0]}")


def build_preprocessor() -> ColumnTransformer:
    """CRIAR UM PREPROCESSAMENTO PARA AS COLUNAS"""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    categories=[PURPOSE_CATEGORIES],
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            )
        ]
    )

    # COLUMN TRANSFORMER (NÚMERO E CATEGÓRICO)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # RETORNA
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """RETORNAR OS NOMES DAS FEATURES ORDENADAS APÓS A TRANFORMAÇÃO"""
    categorial_encoder = preprocessor.named_transformers_["cat"]["encoder"]
    categorical_names = list(
        categorial_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )

    # RETORNA
    return NUMERIC_FEATURES + categorical_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, path)
    logger.info(f"Pré-processador salvo em: {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    logger.info(f"Carregando pré-processador de: {path}")

    # RETORNA
    return joblib.load(path)


def prepare_input(dados: dict) -> pd.DataFrame:
    """CONVERTER DICIONÁRIO DE PREVISÃO UM DATAFRAME"""

    # RETORNA
    return pd.DataFrame([dados])
