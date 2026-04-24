from typing import Literal
from pydantic import BaseModel, Field


class LoanInput(BaseModel):
    credit_policy: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 se o cliente estiver apto ao crédito, 0 caso contrário",
        alias="credit.policy",
    )
    purpose: Literal[
        "all_other",
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
    ] = Field(..., description="Propósito do empréstimo")
    int_rate: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Taxa de Juros (ex: 0.11 é 11%).",
        alias="int.rate",
    )
    installment: float = Field(..., gt=0, description="Parcelas mensais")
    log_annual_inc: float = Field(
        ...,
        description="Log natural da renda anual",
        alias="log.annual.inc",
    )
    dti: float = Field(..., ge=0, description="Rendimento do tomador do empréstimo")
    fico: int = Field(..., ge=300, le=850, description="Pontuação de crédito FICO.")
    days_with_cr_line: float = Field(
        ...,
        ge=0,
        description="O número de dias em que o mutuário teve uma linha de crédito",
        alias="days.with.cr.line",
    )
    revol_bal: float = Field(
        ...,
        ge=0,
        description="Saldo rotativo do mutuário (montante não pago no final do ciclo de cobrança do cartão de crédito)",
        alias="revol.bal",
    )
    revol_util: float = Field(
        ...,
        ge=0,
        le=100,
        description="Taxa de utilização da linha rotativa do mutuário (%).",
        alias="revol.util",
    )
    inq_last_6mths: int = Field(
        ...,
        ge=0,
        description="Número de consultas do mutuário por credores nos últimos 6 meses.",
        alias="inq.last.6mths",
    )
    delinq_2yrs: int = Field(
        ...,
        ge=0,
        description="Número de vezes que o mutuário havia passado mais de 30 dias em um pagamento nos últimos 2 anos",
        alias="delinq.2yrs",
    )
    pub_rec: int = Field(
        ...,
        ge=0,
        description="O número de registros públicos depreciativos do mutuário",
        alias="pub.rec",
    )

    model_config = {"populate_by_name": True}

    def to_raw_dict(self) -> dict:
        """RETORNA O DICIONÁRIO COM OS NOMES ORIGINAIS DAS COLUNAS"""
        return {
            "credit.policy": self.credit_policy,
            "purpose": self.purpose,
            "int.rate": self.int_rate,
            "installment": self.installment,
            "log.annual.inc": self.log_annual_inc,
            "dti": self.dti,
            "fico": self.fico,
            "days.with.cr.line": self.days_with_cr_line,
            "revol.bal": self.revol_bal,
            "revol.util": self.revol_util,
            "inq.last.6mths": self.inq_last_6mths,
            "delinq.2yrs": self.delinq_2yrs,
            "pub.rec": self.pub_rec,
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Fully Paid, 1 = Default")
    probability: float = Field(..., description="Probability of default (0–1)")
    risk_label: str = Field(..., description="Human-readable risk band")
    label: str = Field(..., description="'Fully Paid' or 'Default'")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
