# 💳 Previsão de Inadimplência de Empréstimos

[![CI Pipeline](https://github.com/valensoela/analise-credito-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/valensoela/analise-credito-ml/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Sistema de Machine Learning end-to-end que prevê se um tomador de empréstimo da LendingClub irá **pagar integralmente** ou **inadimplir** — da análise exploratória a uma API REST pronta para produção com stack completa de MLOps.

---

## 📋 Índice

- [Contexto de Negócio](#-contexto-de-negócio)
- [Arquitetura](#️-arquitetura)
- [Dataset](#-dataset)
- [Modelos e Resultados](#-modelos-e-resultados)
- [Stack de Tecnologias](#️-stack-de-tecnologias)
- [Como Executar](#-como-executar)
- [Docker](#-docker-setup-completo-em-um-comando)
- [Uso da API](#-uso-da-api)
- [MLflow](#-rastreamento-de-experimentos-com-mlflow)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Decisões de Design](#-decisões-de-design)

---

## 🏦 Contexto de Negócio

Instituições financeiras perdem bilhões anualmente com inadimplência. Identificar tomadores de alto risco **antes** de conceder um empréstimo é uma das aplicações de maior impacto do Machine Learning no setor de crédito.

Este projeto constrói um sistema completo de ML treinado com dados históricos do [LendingClub](https://www.lendingclub.com/) para:
- Prever se um tomador vai **pagar integralmente** ou **inadimplir**
- Retornar uma **probabilidade de inadimplência** e um **rótulo de risco** legível
- Expor as predições via **API REST** pronta para integração com outros sistemas

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                       docker-compose                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────┐    ┌─────────┐  │
│  │  MLflow  │──> │  Train   │──> │  API   │──> │Streamlit│  │
│  │  :5000   │    │          │    │ :8000  │    │  :8501  │  │
│  └──────────┘    └──────────┘    └────────┘    └─────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Todo o stack sobe automaticamente com um único comando. O container de treinamento roda primeiro, loga todos os experimentos no MLflow, salva o melhor modelo e somente então a API e a demo ficam disponíveis.

---

## 📊 Dataset

Dados históricos de empréstimos do LendingClub com **14 variáveis**:

| Variável | Descrição |
|---|---|
| `credit.policy` | Atende aos critérios de subscrição do LendingClub (1/0) |
| `purpose` | Finalidade do empréstimo (cartão de crédito, consolidação de dívida, etc.) |
| `int.rate` | Taxa de juros — maior taxa indica maior risco |
| `installment` | Valor da parcela mensal (R$) |
| `log.annual.inc` | Log da renda anual autodeclarada do tomador |
| `dti` | Relação dívida/renda |
| `fico` | Pontuação de crédito FICO (300–850) |
| `days.with.cr.line` | Dias com linha de crédito ativa |
| `revol.bal` | Saldo rotativo (R$) |
| `revol.util` | Taxa de utilização da linha rotativa (%) |
| `inq.last.6mths` | Consultas de credores nos últimos 6 meses |
| `delinq.2yrs` | Atrasos acima de 30 dias nos últimos 2 anos |
| `pub.rec` | Registros públicos negativos (falências, penhoras) |
| **`not.fully.paid`** | **Alvo: 1 = Inadimplente, 0 = Pagou integralmente** |

---

## 🤖 Modelos e Resultados

Três classificadores treinados com tratamento de desbalanceamento de classes (`scale_pos_weight` / `class_weight="balanced"`):

| Modelo | ROC-AUC | F1 (Inadimplente) | Recall (Inadimplente) | Precisão (Inadimplente) |
|---|---|---|---|---|
| Decision Tree | **0.50** | **0.163** | **0.166** | **0.160** |
| Random Forest | **0.65** | **0.26** | **0.25** | **0.27** |
| **XGBoost** ⭐ | **0.634** | **0.310** | **0.469** | **0.232** |

> O **XGBoost** foi selecionado como melhor modelo com base no ROC-AUC e registrado no MLflow Model Registry.

**Por que essas métricas importam no crédito:**
Na previsão de inadimplência, um **Falso Negativo** (prever "Vai Pagar" quando o tomador inadimple) é muito mais custoso do que um Falso Positivo. O XGBoost usa um **threshold de classificação de 0.40** (em vez do padrão 0.50) para maximizar o recall na classe de inadimplentes — detectando mais casos de risco ao custo de alguns alarmes extras.

---

## 🛠️ Stack de Tecnologias

| Camada | Tecnologia | Função |
|---|---|---|
| **Linguagem** | Python 3.11 | — |
| **Gerenciador de Pacotes** | UV | Gerenciamento moderno e rápido de dependências |
| **ML** | scikit-learn · XGBoost · SHAP | Treinamento, avaliação e explicabilidade |
| **MLOps** | MLflow | Rastreamento de experimentos + Model Registry |
| **API** | FastAPI + Pydantic | API REST de produção com validação |
| **Demo** | Streamlit + Plotly | Interface web interativa |
| **Containers** | Docker + Docker Compose | Deploy reproduzível |
| **CI/CD** | GitHub Actions | Testes e lint automatizados |
| **Testes** | pytest + pytest-cov | Testes unitários com cobertura |
| **Linting** | Ruff | Verificador de qualidade de código |

---

## 🚀 Como Executar

### Opção 1 — Docker (recomendado)

Zero configuração manual. Um comando sobe todo o stack:

```bash
git clone https://github.com/valensoela/analise-credito-ml.git
cd analise-credito-ml
```

Adicione o dataset em `data/raw/dados.csv` e execute:

```bash
docker compose up --build
```

Isso vai automaticamente:
1. Subir o servidor de rastreamento MLflow
2. Treinar os 3 modelos e logar os experimentos
3. Registrar o melhor modelo no Model Registry
4. Subir a API REST
5. Subir a demo interativa

| Serviço | URL |
|---|---|
| 🔬 MLflow UI | http://localhost:5000 |
| ⚡ Documentação da API | http://localhost:8000/docs |
| 🎯 Demo Interativa | http://localhost:8501 |

---

### Opção 2 — Local com UV

**Pré-requisitos:** Python 3.11, [UV](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/valensoela/analise-credito-ml.git
cd analise-credito-ml

# Instalar dependências
uv sync

# Subir o MLflow (manter este terminal aberto)
uv run mlflow ui --host 127.0.0.1 --port 5000

# Treinar os modelos (novo terminal)
uv run python src/train.py --model all

# Subir a API (novo terminal)
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Subir a demo (novo terminal)
uv run streamlit run app/streamlit_app.py
```

Ou use os atalhos do Makefile:
```bash
make mlflow    # terminal 1
make train     # terminal 2
make api       # terminal 3
make streamlit # terminal 4
```

---

## 🔌 Uso da API

### Health Check

```bash
curl http://localhost:8000/health
```

### Predição Individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "pub.rec": 0
  }'
```

**Resposta:**
```json
{
  "prediction": 0,
  "probability": 0.1823,
  "risk_label": "Low Risk",
  "label": "Fully Paid"
}
```

### Predição em Lote

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{ ...emprestimo1... }, { ...emprestimo2... }]'
```

Documentação interativa completa disponível em `http://localhost:8000/docs` (Swagger UI).

---

## 📈 Rastreamento de Experimentos com MLflow

Cada execução de treinamento é completamente registrada:

- **Parâmetros** — todos os hiperparâmetros por modelo
- **Métricas** — ROC-AUC, F1, Precisão, Recall
- **Artefatos** — matriz de confusão, curva ROC, gráfico de importância SHAP
- **Model Registry** — melhor modelo promovido automaticamente

Acesse `http://localhost:5000` para comparar execuções e inspecionar artefatos.

---

## 📁 Estrutura do Projeto

```
analise-credito-ml/
│
├── data/
│   ├── raw/                        # Dataset CSV original
│   └── processed/                  # Dados transformados
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # Análise exploratória
│
├── src/                            # Código-fonte modular
│   ├── data_processing.py          # Pipeline de pré-processamento sklearn
│   ├── train.py                    # Script de treinamento com MLflow
│   ├── evaluate.py                 # Métricas, plots e logging MLflow
│   └── predict.py                  # Classe de inferência
│
├── api/                            # API REST
│   ├── main.py                     # App FastAPI + endpoints
│   └── schemas.py                  # Modelos Pydantic de entrada/saída
│
├── app/
│   └── streamlit_app.py            # Demo interativa
│
├── models/                         # Modelos serializados (gerados)
├── reports/figures/                # Gráficos de treinamento (gerados)
├── tests/                          # Testes unitários
│
├── .github/workflows/ci.yml        # CI com GitHub Actions
├── docker-compose.yml              # Orquestração do stack completo
├── Dockerfile                      # Build multi-stage com UV
├── Makefile                        # Atalhos para desenvolvimento
├── pyproject.toml                  # Dependências (UV)
└── config.yaml                     # Hiperparâmetros e configurações
```

---

## 💡 Decisões de Design

**Threshold ajustado para 0.40**
O threshold padrão de 0.50 foi reduzido para 0.40 no XGBoost. Em risco de crédito, um Falso Negativo (deixar passar um inadimplente) é significativamente mais caro do que um Falso Positivo. Esse ajuste aumenta o recall na classe de inadimplentes de ~38% para ~47%.

**Pipeline sklearn para pré-processamento**
Todo o pré-processamento (StandardScaler + OneHotEncoder) é encapsulado em um `ColumnTransformer` e serializado junto com o modelo. Isso previne vazamento de dados e garante que as mesmas transformações sejam aplicadas na inferência.

**MLflow como fonte da verdade**
Cada execução de treinamento é completamente reproduzível. Parâmetros, métricas e artefatos são logados automaticamente, e o melhor modelo é promovido para o Model Registry — a API sempre carrega o modelo campeão registrado.

**Workflow Docker-first**
Todo o pipeline — treinamento, serving e monitoramento — roda dentro do Docker. Qualquer desenvolvedor pode clonar o repositório, adicionar o dataset em `data/raw/` e executar `docker compose up` para ter um sistema funcional completo sem nenhuma configuração manual.

---

## 🧪 Testes

```bash
make test
# ou
uv run pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE).

---

*Desenvolvido como projeto de portfólio demonstrando engenharia de ML end-to-end — da análise exploratória ao deploy containerizado com observabilidade MLOps completa.*