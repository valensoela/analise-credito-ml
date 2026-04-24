import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Previsor de Inadimplência de Empréstimos", page_icon="💳", layout="wide"
)

# STYLES
st.markdown(
    """
<style>
    .risk-badge {
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        display: inline-block;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# SIDEBAR
with st.sidebar:
    st.header("📋 Solicitação de Empréstimo")

    credit_policy = st.selectbox(
        "Atende aos critérios do LendingClub?",
        options=[1, 0],
        format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
    )

    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "debt_consolidation",
            "credit_card",
            "small_business",
            "major_purchase",
            "educational",
            "home_improvement",
            "all_other",
        ],
    )

    fico = st.slider("FICO Score", min_value=300, max_value=850, value=700, step=5)
    int_rate = st.slider(
        "Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.1
    )
    installment = st.number_input(
        "Monthly Installment ($)", min_value=10.0, max_value=2000.0, value=400.0
    )
    dti = st.slider(
        "Debt-to-Income Ratio", min_value=0.0, max_value=35.0, value=15.0, step=0.5
    )
    log_annual_inc = st.slider(
        "Log Annual Income", min_value=7.0, max_value=14.5, value=11.0, step=0.1
    )
    days_with_cr_line = st.number_input(
        "Days with Credit Line", min_value=0.0, max_value=20000.0, value=3600.0
    )
    revol_bal = st.number_input(
        "Revolving Balance ($)", min_value=0.0, max_value=200000.0, value=12000.0
    )
    revol_util = st.slider(
        "Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=40.0
    )
    inq_last_6mths = st.number_input(
        "Inquiries (last 6 months)", min_value=0, max_value=15, value=1
    )
    delinq_2yrs = st.number_input(
        "Delinquencies (last 2 years)", min_value=0, max_value=20, value=0
    )
    pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)

    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")


# CONTEÚDO
RISK_COLORS = {
    "Risco Muito Baixo": "#2ecc71",
    "Risco Baixo": "#27ae60",
    "Risco Moderado": "#f39c12",
    "Risco Alto": "#e67e22",
    "Risco Muito Alto": "#e74c3c",
}

if predict_btn:
    payload = {
        "credit.policy": credit_policy,
        "purpose": purpose,
        "int.rate": int_rate / 100,
        "installment": installment,
        "log.annual.inc": log_annual_inc,
        "dti": dti,
        "fico": fico,
        "days.with.cr.line": days_with_cr_line,
        "revol.bal": revol_bal,
        "revol.util": revol_util,
        "inq.last.6mths": inq_last_6mths,
        "delinq.2yrs": delinq_2yrs,
        "pub.rec": pub_rec,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        result = response.json()

        prob = result["probability"]
        label = result["label"]
        risk = result["risk_label"]
        color = RISK_COLORS.get(risk, "#888")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Veredito", label, delta=None)

        with col2:
            st.metric("Probabilidade Padrão", f"{prob * 100:.1f}%")

        with col3:
            st.markdown(
                f'<div class="risk-badge" style="background:{color};color:white">{risk}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # GRÁFICO DE MEDIDOR
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Default Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 20], "color": "#d5f5e3"},
                        {"range": [20, 40], "color": "#a9dfbf"},
                        {"range": [40, 60], "color": "#fdebd0"},
                        {"range": [60, 80], "color": "#fad7a0"},
                        {"range": [80, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 40,
                    },
                },
                number={"suffix": "%", "font": {"size": 36}},
            )
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Não foi possível se conectar a API")

else:
    # BEM-VINDO
    st.info("👈 Preencha os dados e clique em **Predict**.", icon="💡")

    st.markdown("### 📊 Sobre o Projeto")
    st.markdown(
        """
        | Modelo |
        |---|
        | Decision Tree |
        | Random Forest |
        | XGBoost |

        **Principais características utilizadas na previsão:**
        - Pontuação FICO (score do crédito)
        - Taxa de juros (prêmio de risco)
        - Índice de endividamento
        - Taxa de utilização rotativa
        - Número de consultas de crédito recentes

        **MLOps Stack:** MLflow · FastAPI · Streamlit · Docker · GitHub Actions
        """
    )
