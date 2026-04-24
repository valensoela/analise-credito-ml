import logging
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """COMPUTAR DICIONÁRIO DE MÉTRICAS AVALIATIVAS"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1),
        "precision_class_1": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    # VERIFICA SE ESTÁ SENDO EXIGIDO Y_PROBA
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # RETORNA
    return metrics


def log_metrics_to_mlflow(metrics: dict) -> None:
    """LOG DE MÉTRICAS PARA O MLFLOW"""
    mlflow.log_metrics(metrics)

    logger.info(f"Log de métricas salvo no MLflow: {metrics}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """CRIAR GRÁFICO DE MATRIZ DE CONFUSÃO"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    # CRIAR HEATMAP
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Paid", "Default"],
        yticklabels=["Paid", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Atual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    # CASO DESEJE SALVAR LOCALMENTE
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=120)
        logger.info(f"Confusion Matrix salva em: {save_path}")

    # RETORNA
    return fig


def plot_roc_curve(y_true, y_proba, model_name: str, save_path: str) -> None:
    """CRIAR GRÁFICO ROC CURVE"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Taxa Falso Positivo")
    ax.set_ylabel("Taxa Verdadeiro Positivo")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    plt.tight_layout()

    # CASO DESEJE SALVAR LOCALMENTE
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=120)
        logger.info(f"ROC Cruve salvo em: {save_path}")

    # RETORNA
    return fig


def generate_report(y_true, y_pred, model_name: str) -> str:
    """RETORNAR RELATÓRIO DE CLASSIFICAÇÃO DO MODELO FORMATADO"""
    report = classification_report(
        y_true, y_pred, target_names=["Fully Paid", "Default"]
    )
    logger.info(f"\n{'='*50}\n{model_name}\n{report}")

    # RETORNAR
    return report
