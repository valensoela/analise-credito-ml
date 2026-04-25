import os
import argparse
import logging
import joblib
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from data_processing import (
    load_data,
    validate_data,
    build_preprocessor,
    save_preprocessor,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)
from evaluate import (
    compute_metrics,
    log_metrics_to_mlflow,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_report,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """CARREGAR CONFIGURAÇÕES YAML"""
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(name: str, cfg: dict, scale_pos_weight: float = 1.0):
    """INTANCIANDO MODEL COM AS CONFIGURAÇÕES"""
    if name == "decision_tree":
        return DecisionTreeClassifier(**cfg["models"]["decision_tree"])

    elif name == "random_forest":
        return RandomForestClassifier(**cfg["models"]["random_forest"])

    elif name == "xgboost":
        params = cfg["models"]["xgboost"].copy()
        threshold = params.pop("threshold", 0.4)
        params["scale_pos_weight"] = scale_pos_weight

        return XGBClassifier(**params), threshold

    raise ValueError(f"Modelo desconhecido: {name}")


def train_and_log(
    model_name: str,
    model,
    x_train,
    x_test,
    y_train,
    y_test,
    feature_names: list,
    cfg: dict,
    threshold: float = 0.5,
) -> dict:
    """TREINAR MODELO SALVANDO MÉTRICAS NO MLFLOW"""
    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"Treinando modelo: {model_name}...")

        # TAG
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "LendingClub")

        # LOG HIPER PARÂMETROS
        mlflow.log_params(model.get_params())

        if threshold != 0.5:
            mlflow.log_param("threshold", threshold)

        # TREINAMENTO
        model.fit(x_train, y_train)

        # PREDICT
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

        else:
            y_proba = None
            y_pred = model.predict(x_test)

        # MÉTRICAS
        metrics = compute_metrics(y_test, y_pred, y_proba)
        log_metrics_to_mlflow(metrics)
        generate_report(y_test, y_pred, model_name)

        # PLOTS
        cm_path = f"reports/figures/{model_name}_confusion_matrix.png"
        fig_cm = plot_confusion_matrix(y_test, y_pred, model_name, cm_path)

        mlflow.log_artifact(cm_path)
        plt.close(fig_cm)

        if y_proba is not None:
            roc_path = f"reports/figures/{model_name}_roc_curve.png"
            fig_roc = plot_roc_curve(y_test, y_proba, model_name, roc_path)

            mlflow.log_artifact(roc_path)
            plt.close(fig_roc)

        # SHAP
        if model_name == "xgboost":
            log_shap(model, x_test, feature_names)

        # MODEL
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # OBTÉM O ID DE EXECUÇÃO
        run_id = run.info.run_id
        logger.info(f"{model_name} -> run_id={run_id} | metrics={metrics}")

    # RETORNA
    return {"run_id": run_id, "metrics": metrics, "model": model}


def log_shap(model, x_test, feature_names):
    """COMPUTAR E SALVAR REUMO DO GRÁFICO SHAP"""
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(x_test)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values, show=False)

        shap_path = "reports/figures/xgboost_shap_importance.png"
        Path(shap_path).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(shap_path, dpi=120)
        mlflow.log_artifact(shap_path)

        plt.close()
        logger.info("Log de feature importance (SHAP) salvo")

    except Exception as e:
        logger.warning(f"Erro Gráfico SHAP: {e}")


def promote_best_model(results: dict, cfg: dict) -> None:
    """REGISTRAR O MELHOR MODELO NO MLFLOW"""
    best_name = max(
        results,
        key=lambda k: results[k]["metrics"].get(
            "roc_auc", results[k]["metrics"]["f1_macro"]
        ),
    )
    best_run_id = results[best_name]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    registry_name = cfg["mlflow"]["model_registry_name"]

    logger.info(
        f"Promovendo '{best_name}' (run={best_run_id}) -> registry '{registry_name}'"
    )
    mlflow.register_model(model_uri, registry_name)


def save_best_model_locally(results: dict, preprocessor, cfg: dict) -> None:
    """PERSISTIR O MELHOR MODELO EM ARQUIVO LOCAL"""
    best_name = max(
        results,
        key=lambda k: results[k]["metrics"].get(
            "roc_auc", results[k]["metrics"]["f1_macro"]
        ),
    )
    best_model = results[best_name]["model"]
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, models_dir / "best_model.joblib")
    save_preprocessor(preprocessor, str(models_dir / "preprocessor.joblib"))

    # SALVAR MODELOS DE FORMA INDIVIDUAL
    for name, res in results.items():
        joblib.dump(res["model"], models_dir / f"{name}.joblib")

    logger.info(f"Melhor modelo ('{best_name}') salvo em models/best_model.joblib")


def main(args):
    cfg = load_config(args.config)

    # CARREGAR E VALIDAR
    df = load_data(cfg["data"]["raw_path"])
    validate_data(df)

    x_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # PRÉ-PROCESSAMENTO
    preprocessor = build_preprocessor()

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)
    features_names = get_feature_names(preprocessor)

    # MFLOW SETUP
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    )

    # TREINAMENTO
    models_to_train = (
        ["decision_tree", "random_forest", "xgboost"]
        if args.model == "all"
        else [args.model]
    )

    scale_pos_weight = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    results = {}

    # PARA CADA MODELO
    for model_name in models_to_train:
        threshold = 0.5

        # XGBOOST
        if model_name == "xgboost":
            model, threshold = get_model(model_name, cfg, scale_pos_weight)

        # DEMAIS MODELOS
        else:
            model = get_model(model_name, cfg)

        results[model_name] = train_and_log(
            model_name=model_name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=features_names,
            cfg=cfg,
            threshold=threshold,
        )

    # PROMOVER E SALVAR
    if len(results) > 1:
        promote_best_model(results, cfg)

    save_best_model_locally(results, preprocessor, cfg)

    # LOG COMPARATIVO
    logger.info("\nTreinamento completo\n")
    logger.info("Comparação de modelo:")

    for name, res in results.items():
        m = res["metrics"]

        logger.info(
            f"  {name:<20} AUC={m.get('roc_auc', 'N/A'):.3f}  "
            f"F1={m['f1_class_1']:.3f}  "
            f"Recall={m['recall_class_1']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modelos de previsão de inadimplência de empréstimos"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "decision_tree", "random_forest", "xgboost"],
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args)
