import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================
# 1) CARREGAR PREÇOS + DIVIDENDOS (Yahoo – data EX)
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    ativo = yf.Ticker(ticker)

    # Preços históricos (não ajustados)
    hist = ativo.history(period="2y", auto_adjust=False)
    precos = hist["Close"].dropna()

    # Garantir índice sem timezone
    try:
        precos.index = precos.index.tz_localize(None)
    except Exception:
        pass

    # Dividendos (data ex)
    dividendos = ativo.dividends
    try:
        dividendos.index = dividendos.index.tz_localize(None)
    except Exception:
        pass

    return precos, dividendos


# ============================================================
# 2) BACKTEST COLLAR
# ============================================================

def backtest_collar(
    precos: pd.Series,
    dividendos: pd.Series,
    prazo_du: int,
    ganho_max: float,
    perda_max: float,
    risk_free: float,
    dias_ano: int = 252,
):

    datas = precos.index

    if len(datas) <= prazo_du:
        return None

    # Retorno da ação
    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    # Dividendos dentro da janela
    ret_div = []
    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]
        soma = 0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

    ret_div = np.array(ret_div)

    # Payoff Collar
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    # Estrutura favorável?
    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)

    # Retorno final (com dividendos)
    ret_op_com_div = ret_op_sem_div + ret_div

    # Bateu CDI?
    rent_anual_op = (1 + ret_op_com_div) ** (dias_ano / prazo_du) - 1
    bate_cdi = (rent_anual_op > risk_free).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_sem_div": ret_op_sem_div,
        "ret_op_com_div": ret_op_com_div,
        "deu_certo": deu_certo,
        "bate_cdi": bate_cdi,
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
    }

    return df, resumo, dividendos


# ============================================================
# 3) GRÁFICO – Collar x IBOV (não acumulado)
# ============================================================

def gerar_grafico(df: pd.DataFrame, ticker: str) -> BytesIO:

    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]

    try:
        ibov.index = ibov.index.tz_localize(None)
    except:
        pass

    ibov_ret = []
    for i in range(len(df)):
        ini = df.iloc[i]["data_inicio"]
        fim = df.iloc[i]["data_fim"]
        if ini in ibov.index and fim in ibov.index:
            ibov_ret.append(ibov.loc[fim] / ibov.loc[ini] - 1)
        else:
            ibov_ret.append(np.nan)

    df_plot = df.copy()
    df_plot["ret_ibov"] = ibov_ret

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Retornos por operação – Collar x IBOV (Não acumulado)", fontsize=14, weight="bold")
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno no período")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# 4) INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="Backtest Collar Profissional", layout="wide")

st.title("📈 Backtest Collar – Dividendos (Yahoo Data EX)")
st.markdown("Preços reais + dividendos (data ex) + CDI anualizado + análise da eficácia da estrutura.")

# -----------------------------
# Inputs 100% editáveis
# -----------------------------
st.sidebar.header("Configurações")

ticker = st.sidebar.text_input("Ticker do ativo:", "EZTC3.SA")

prazo_du = st.sidebar.number_input(
    "Prazo da operação (dias úteis):",
    value=63,
    step=1,
    format="%d"
)

ganho_max = st.sidebar.number_input(
    "Ganho máximo (%):",
    value=8.0,
    step=0.1,
    format="%.2f"
) / 100

perda_max = st.sidebar.number_input(
    "Perda máxima (%):",
    value=8.0,
    step=0.1,
    format="%.2f"
) / 100

risk_free = st.sidebar.number_input(
    "CDI / Risk-free anual (%):",
    value=15.0,
    step=0.1,
    format="%.2f"
) / 100

rodar = st.sidebar.button("🚀 Rodar Backtest")

# -----------------------------
# Execução
# -----------------------------
if rodar:

    precos, dividendos = carregar_preco_e_dividendos(ticker)
    resultado = backtest_collar(precos, dividendos, prazo_du, ganho_max, perda_max, risk_free)

    if resultado is None:
        st.error("Histórico insuficiente para esse prazo.")
    else:
        df, resumo, dividendos = resultado

        # Métricas principais
        col1, col2 = st.columns(2)
        col1.metric("Estrutura Favorável (%)", f"{resumo['pct_deu_certo']*100:.1f}%")
        col2.metric("Bateu CDI (%)", f"{resumo['pct_bate_cdi']*100:.1f}%")

        # Dividendos
        st.subheader("📌 Dividendos (data EX – Yahoo)")
        if dividendos.empty:
            st.warning("Nenhum dividendo encontrado no período.")
        else:
            st.dataframe(dividendos.rename("valor_por_acao"))

        # Gráfico
        graf = gerar_grafico(df, ticker)
        st.image(graf, caption="Retornos por operação – Collar x IBOV")

        # Tabela final
        st.subheader("📄 Detalhamento das operações")
        st.dataframe(df)
