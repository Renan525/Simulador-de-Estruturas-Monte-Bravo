import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from math import log, sqrt, exp, erf


# ============================================================
# FUNÇÕES AUXILIARES GERAIS
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    """Carrega preços (Close) e dividendos (data ex) do Yahoo."""
    ativo = yf.Ticker(ticker)

    hist = ativo.history(period="2y", auto_adjust=False)
    precos = hist["Close"].dropna()

    try:
        precos.index = precos.index.tz_localize(None)
    except Exception:
        pass

    dividendos = ativo.dividends

    try:
        dividendos.index = dividendos.index.tz_localize(None)
    except Exception:
        pass

    return precos, dividendos


def gerar_ret_ibov(df_datas: pd.DataFrame):
    """Gera série de retornos do IBOV na mesma janela de datas do df (data_inicio/data_fim)."""
    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]

    try:
        ibov.index = ibov.index.tz_localize(None)
    except Exception:
        pass

    ibov_ret = []
    for i in range(len(df_datas)):
        ini = df_datas.iloc[i]["data_inicio"]
        fim = df_datas.iloc[i]["data_fim"]
        if ini in ibov.index and fim in ibov.index:
            ibov_ret.append(ibov.loc[fim] / ibov.loc[ini] - 1)
        else:
            ibov_ret.append(np.nan)
    return np.array(ibov_ret)


# ============================================================
# FUNÇÕES – COLLAR
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

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    # Dividendos na janela
    ret_div = []
    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]
        soma = 0.0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

    ret_div = np.array(ret_div)

    # Payoff da collar
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0.0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0.0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    # Estrutura favorável: proteção atuou OU não houve limitação de ganho
    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)

    # Retorno com dividendos (para CDI)
    ret_op_com_div = ret_op_sem_div + ret_div
    rent_anual_op = (1 + ret_op_com_div) ** (dias_ano / prazo_du) - 1
    bate_cdi = (rent_anual_op > risk_free).astype(int)

    df = pd.DataFrame(
        {
            "data_inicio": datas[:-prazo_du],
            "data_fim": datas[prazo_du:],
            "ret_preco": ret_preco,
            "ret_dividendos": ret_div,
            "ret_op_sem_div": ret_op_sem_div,
            "ret_op_com_div": ret_op_com_div,
            "deu_certo": deu_certo,
            "bate_cdi": bate_cdi,
        }
    )

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df: pd.DataFrame, ticker: str) -> BytesIO:
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Retornos por operação – Collar x IBOV (não acumulado)", fontsize=14, weight="bold")
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno no período")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ------------------------------------------------------------
# ABA AP – ALOCAÇÃO PROTEGIDA
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Estratégia Alocação Protegida (AP)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – AP**")

    ticker_ap = st.sidebar.text_input("Ticker (AP):", "EZTC3.SA", key="ticker_ap")

    prazo_du_ap = st.sidebar.number_input(
        "Prazo da operação (dias úteis) – AP:",
        value=63,
        step=1,
        format="%d",
        key="prazo_ap"
    )

    perda_max_ap = st.sidebar.number_input(
        "Perda máxima protegida (%):",
        value=5.0,
        step=0.1,
        format="%.2f",
        key="perda_ap"
    ) / 100

    risk_free_ap = st.sidebar.number_input(
        "CDI / Risk-free anual (%):",
        value=15.0,
        step=0.1,
        format="%.2f",
        key="cdi_ap"
    ) / 100

    rodar_ap = st.sidebar.button("🚀 Rodar AP", key="rodar_ap")

    if rodar_ap:
        # -------------------------------
        # 1. Preços + dividendos
        # -------------------------------
        precos_ap, dividendos_ap = carregar_preco_e_dividendos(ticker_ap)
        resultado_ap = backtest_ap(precos_ap, dividendos_ap, prazo_du_ap, perda_max_ap, risk_free_ap)

        if resultado_ap is None:
            st.error("Histórico insuficiente para esse prazo na aba AP.")
        else:
            df_ap, resumo_ap, dividendos_ap = resultado_ap

            # -------------------------------
            # 2. Cálculo DO PREÇO JUSTO HOJE
            # -------------------------------
            S0_atual = precos_ap.iloc[-1]
            K_atual = S0_atual * (1 - perda_max_ap)
            sigma_anual = resumo_ap["vol_anual"]
            T = prazo_du_ap / 252

            preco_put_hoje = black_scholes_put(
                S0=S0_atual,
                K=K_atual,
                r=risk_free_ap,
                sigma=sigma_anual,
                T=T
            )

            custo_pct_hoje = preco_put_hoje / S0_atual  # porcentagem do ativo

            # -------------------------------
            # 3. MÉTRICAS
            # -------------------------------
            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Estrutura Favorável (%)", f"{resumo_ap['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu CDI (%)", f"{resumo_ap['pct_bate_cdi']*100:.1f}%")

            col3.metric("Preço Justo Atual (R$)", f"R$ {preco_put_hoje:.4f}")
            col4.metric("Custo Atual da Put (% do ativo)", f"{custo_pct_hoje*100:.2f}%")

            col5.metric("Vol anual usada", f"{sigma_anual*100:.1f}%")

            # -------------------------------
            # 4. Dividendos (debug)
            # -------------------------------
            st.subheader("📌 Dividendos (data EX – Yahoo) – AP")
            if dividendos_ap.empty:
                st.warning("Nenhum dividendo encontrado no período.")
            else:
                st.dataframe(dividendos_ap.rename("valor_por_acao"))

            # -------------------------------
            # 5. Preço justo da put por operação (histórico)
            # -------------------------------
            st.subheader("📘 Preço justo da Put (BSL) por operação no backtest")
            st.dataframe(df_ap[["data_inicio", "data_fim", "preco_put_bsl"]])

            # -------------------------------
            # 6. Gráfico AP x IBOV
            # -------------------------------
            graf_ap = gerar_grafico_ap(df_ap, ticker_ap)
            st.image(graf_ap, caption="Retornos por operação – AP x IBOV")

            # -------------------------------
            # 7. Detalhamento da operação AP
            # -------------------------------
            st.subheader("📄 Detalhamento – AP")
            st.dataframe(df_ap)

# ============================================================
# STREAMLIT – DASHBOARD UNIFICADO
# ============================================================

st.set_page_config(page_title="Backtest Estruturas – Collar & AP", layout="wide")

st.title("📈 Backtest de Estruturas – Collar & Alocação Protegida")
st.markdown(
    """
    Ambiente unificado para análise de **Collar** e **Alocação Protegida (AP)**,
    com uso de preços reais (Yahoo), dividendos, CDI e precificação BS para puts.
    """
)

tab_collar, tab_ap = st.tabs(["📊 Collar", "🛡️ Alocação Protegida (AP)"])

# ------------------------------------------------------------
# ABA COLLAR
# ------------------------------------------------------------
with tab_collar:
    st.subheader("📊 Estratégia Collar")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – Collar**")

    ticker_c = st.sidebar.text_input("Ticker (Collar):", "EZTC3.SA", key="ticker_c")

    prazo_du_c = st.sidebar.number_input(
        "Prazo da operação (dias úteis) – Collar:",
        value=63,
        step=1,
        format="%d",
        key="prazo_c",
    )

    ganho_max_c = st.sidebar.number_input(
        "Ganho máximo (%):",
        value=8.0,
        step=0.1,
        format="%.2f",
        key="ganho_c",
    ) / 100

    perda_max_c = st.sidebar.number_input(
        "Perda máxima (%):",
        value=8.0,
        step=0.1,
        format="%.2f",
        key="perda_c",
    ) / 100

    risk_free_c = st.sidebar.number_input(
        "CDI / Risk-free anual (%):",
        value=15.0,
        step=0.1,
        format="%.2f",
        key="cdi_c",
    ) / 100

    rodar_c = st.sidebar.button("🚀 Rodar Collar", key="rodar_collar")

    if rodar_c:
        precos_c, dividendos_c = carregar_preco_e_dividendos(ticker_c)
        resultado_c = backtest_collar(precos_c, dividendos_c, prazo_du_c, ganho_max_c, perda_max_c, risk_free_c)

        if resultado_c is None:
            st.error("Histórico insuficiente para esse prazo na aba Collar.")
        else:
            df_c, resumo_c, dividendos_c = resultado_c

            col1, col2 = st.columns(2)
            col1.metric("Estrutura Favorável (%)", f"{resumo_c['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu CDI (%)", f"{resumo_c['pct_bate_cdi']*100:.1f}%")

            st.subheader("📌 Dividendos (data EX – Yahoo) – Collar")
            if dividendos_c.empty:
                st.warning("Nenhum dividendo encontrado no período.")
            else:
                st.dataframe(dividendos_c.rename("valor_por_acao"))

            graf_c = gerar_grafico_collar(df_c, ticker_c)
            st.image(graf_c, caption="Retornos por operação – Collar x IBOV")

            st.subheader("📄 Detalhamento – Collar")
            st.dataframe(df_c)

# ------------------------------------------------------------
# ABA AP
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Estratégia Alocação Protegida (AP)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – AP**")

    ticker_ap = st.sidebar.text_input("Ticker (AP):", "EZTC3.SA", key="ticker_ap")

    prazo_du_ap = st.sidebar.number_input(
        "Prazo da operação (dias úteis) – AP:",
        value=63,
        step=1,
        format="%d",
        key="prazo_ap",
    )

    perda_max_ap = st.sidebar.number_input(
        "Perda máxima protegida (%):",
        value=5.0,
        step=0.1,
        format="%.2f",
        key="perda_ap",
    ) / 100

    risk_free_ap = st.sidebar.number_input(
        "CDI / Risk-free anual (%):",
        value=15.0,
        step=0.1,
        format="%.2f",
        key="cdi_ap",
    ) / 100

    rodar_ap = st.sidebar.button("🚀 Rodar AP", key="rodar_ap")

    if rodar_ap:
        precos_ap, dividendos_ap = carregar_preco_e_dividendos(ticker_ap)
        resultado_ap = backtest_ap(precos_ap, dividendos_ap, prazo_du_ap, perda_max_ap, risk_free_ap)

        if resultado_ap is None:
            st.error("Histórico insuficiente para esse prazo na aba AP.")
        else:
            df_ap, resumo_ap, dividendos_ap = resultado_ap

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Estrutura Favorável (%)", f"{resumo_ap['pct_deu_certo']*100:.1f}%")
            col2.metric("Bateu CDI (%)", f"{resumo_ap['pct_bate_cdi']*100:.1f}%")
            col3.metric("Preço Justo Médio da Put", f"R$ {resumo_ap['preco_put_medio']:.4f}")
            col4.metric("Vol anual usada", f"{resumo_ap['vol_anual']*100:.1f}%")

            st.subheader("📌 Dividendos (data EX – Yahoo) – AP")
            if dividendos_ap.empty:
                st.warning("Nenhum dividendo encontrado no período.")
            else:
                st.dataframe(dividendos_ap.rename("valor_por_acao"))

            st.subheader("Preço justo da Put (BSL) por operação")
            st.dataframe(df_ap[["data_inicio", "data_fim", "preco_put_bsl"]])

            graf_ap = gerar_grafico_ap(df_ap, ticker_ap)
            st.image(graf_ap, caption="Retornos por operação – AP x IBOV")

            st.subheader("📄 Detalhamento – AP")
            st.dataframe(df_ap)
