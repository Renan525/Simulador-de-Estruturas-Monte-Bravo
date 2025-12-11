import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from math import log, sqrt, exp, erf
import requests
import datetime

# ============================================================
# PARÂMETROS
# ============================================================

DIAS_ANO = 252


# ============================================================
# FUNÇÕES AUXILIARES – PREÇOS E DIVIDENDOS
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
    ativo = yf.Ticker(ticker)

    hist = ativo.history(period="2y", auto_adjust=False)
    precos = hist["Close"].dropna()

    try:
        precos.index = precos.index.tz_localize(None)
    except:
        pass

    dividendos = ativo.dividends
    try:
        dividendos.index = dividendos.index.tz_localize(None)
    except:
        pass

    return precos, dividendos


def gerar_ret_ibov(df_datas: pd.DataFrame):
    ibov = yf.Ticker("^BVSP").history(period="2y", auto_adjust=False)["Close"]
    try:
        ibov.index = ibov.index.tz_localize(None)
    except:
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
# CDI HISTÓRICO – BACEN (série 12)
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_cdi_com_fator():
    hoje = datetime.date.today()
    data_final = hoje.strftime("%d/%m/%Y")
    data_inicial = (hoje - datetime.timedelta(days=3*365)).strftime("%d/%m/%Y")

    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/"
        f"dados?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()

    df = pd.DataFrame(r.json())
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float) / 100.0  # taxa anual decimal
    df = df.set_index("data").sort_index()

    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1
    return df


def riskfree_periodo(cdi_df: pd.DataFrame, ini, fim):
    idx = pd.date_range(start=ini, end=fim, freq="D")
    serie = cdi_df["fator_diario"].reindex(idx).ffill()
    return (1 + serie).prod() - 1, serie


def obter_r_ano_cdi(cdi_df: pd.DataFrame, data):
    serie = cdi_df["valor"].loc[:data]
    return serie.iloc[-1] if not serie.empty else cdi_df["valor"].iloc[0]


# ============================================================
# BLACK-SCHOLES + VOLATILIDADE
# ============================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def estimar_vol_anual(precos: pd.Series):
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    return log_ret.std() * np.sqrt(DIAS_ANO)


def black_scholes_put(S0, K, r, sigma, T):
    if T <= 0:
        return max(K - S0, 0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return max(K * exp(-r*T) * norm_cdf(-d2) - S0 * norm_cdf(-d1), 0)


def black_scholes_call(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0)
    if sigma <= 0:
        return max(S0 - K * exp(-r * T), 0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return max(S0 * norm_cdf(d1) - K * exp(-r*T) * norm_cdf(d2), 0)


# ============================================================
# COLLAR – BACKTEST
# ============================================================

def backtest_collar(precos, dividendos, cdi_df, prazo_du, ganho_max, perda_max):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div, cdi_periodos, cdi_debug = [], [], []

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum() if not dividendos.empty else 0
        ret_div.append(soma_div / p0[i])

        cdi_acum, serie = riskfree_periodo(cdi_df, ini, fim)
        cdi_periodos.append(cdi_acum)
        cdi_debug.append(serie)

    ret_div = np.array(ret_div)
    cdi_periodos = np.array(cdi_periodos)

    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho)
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)

    ret_op_com_div = ret_op_sem_div + ret_div
    bate_cdi = (ret_op_com_div > cdi_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_sem_div": ret_op_sem_div,
        "ret_op_com_div": ret_op_com_div,
        "cdi_periodo": cdi_periodos,
        "deu_certo": deu_certo,
        "bate_cdi": bate_cdi,
        "cdi_detalhado": cdi_debug
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean()
    }

    return df, resumo, dividendos


# ============================================================
# AP – ALOCAÇÃO PROTEGIDA (com SPREAD da PUT)
# ============================================================

def backtest_ap(precos, dividendos, cdi_df, prazo_du, perda_max, preco_put_cotada):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    # PUT hoje – justa
    sigma_atual = estimar_vol_anual(precos.tail(DIAS_ANO))
    S0_atual = precos.iloc[-1]
    K_atual = S0_atual * (1 - perda_max)
    T_atual = prazo_du / DIAS_ANO
    r_atual = obter_r_ano_cdi(cdi_df, datas[-1])

    preco_put_justa_hoje = black_scholes_put(S0_atual, K_atual, r_atual, sigma_atual, T_atual)
    markup_put = preco_put_cotada / preco_put_justa_hoje if preco_put_justa_hoje > 0 else 1.0

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    preco_put_justo_hist = []
    preco_put_ajustado_hist = []
    custo_put_pct = []
    cdi_periodos, cdi_debug = [], []
    sigma_local_hist = []

    sigma_global = estimar_vol_anual(precos)

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum() if not dividendos.empty else 0
        ret_div.append(soma_div / p0[i])

        hist_pre = precos.loc[:ini].tail(DIAS_ANO)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigma_local_hist.append(sigma_local)

        cdi_acum, serie = riskfree_periodo(cdi_df, ini, fim)
        cdi_periodos.append(cdi_acum)
        cdi_debug.append(serie)

        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / DIAS_ANO
        r_local = obter_r_ano_cdi(cdi_df, ini)

        preco_put_just = black_scholes_put(S0, K, r_local, sigma_local, T)
        preco_put_adj = preco_put_just * markup_put

        preco_put_justo_hist.append(preco_put_just)
        preco_put_ajustado_hist.append(preco_put_adj)
        custo_put_pct.append(preco_put_adj / S0)

    ret_div = np.array(ret_div)
    custo_put_pct = np.array(custo_put_pct)
    cdi_periodos = np.array(cdi_periodos)
    sigma_local_hist = np.array(sigma_local_hist)

    ret_ap_sem_div = ret_preco - custo_put_pct
    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)
    bate_cdi = (ret_ap_com_div > cdi_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "preco_put_justo_hist": preco_put_justo_hist,
        "preco_put_ajustado_hist": preco_put_ajustado_hist,
        "markup_put": markup_put,
        "custo_put_pct": custo_put_pct,
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_ap_sem_div": ret_ap_sem_div,
        "ret_ap_com_div": ret_ap_com_div,
        "cdi_periodo": cdi_periodos,
        "hedge_acionado": hedge_acionado,
        "deu_certo": deu_certo,
        "bate_cdi": bate_cdi,
        "sigma_local": sigma_local_hist,
        "cdi_detalhado": cdi_debug
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
        "preco_put_justa_hoje": preco_put_justa_hoje,
        "preco_put_cotada": preco_put_cotada,
        "markup_put": markup_put
    }

    return df, resumo, dividendos


# ============================================================
# FINANCIAMENTO / COVERED CALL (com SPREAD da CALL)
# ============================================================

def backtest_financiamento(precos, dividendos, cdi_df, prazo_du, ganho_max, preco_call_cotada):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    # CALL hoje – justa
    sigma_atual = estimar_vol_anual(precos.tail(DIAS_ANO))
    S0_atual = precos.iloc[-1]
    K_atual = S0_atual * (1 + ganho_max)
    T_atual = prazo_du / DIAS_ANO
    r_atual = obter_r_ano_cdi(cdi_df, datas[-1])

    preco_call_justa_hoje = black_scholes_call(S0_atual, K_atual, r_atual, sigma_atual, T_atual)
    markup_call = preco_call_cotada / preco_call_justa_hoje if preco_call_justa_hoje > 0 else 1.0

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    call_justa_hist = []
    call_ajustada_hist = []
    premio_call_pct = []
    cdi_periodos, cdi_debug = [], []
    sigma_local_hist = []

    sigma_global = estimar_vol_anual(precos)

    for i in range(len(p0)):
        ini, fim = datas[i], datas[i + prazo_du]

        soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum() if not dividendos.empty else 0
        ret_div.append(soma_div / p0[i])

        hist_pre = precos.loc[:ini].tail(DIAS_ANO)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigma_local_hist.append(sigma_local)

        cdi_acum, serie = riskfree_periodo(cdi_df, ini, fim)
        cdi_periodos.append(cdi_acum)
        cdi_debug.append(serie)

        S0 = p0[i]
        K = S0 * (1 + ganho_max)
        T = prazo_du / DIAS_ANO
        r_local = obter_r_ano_cdi(cdi_df, ini)

        preco_call_just = black_scholes_call(S0, K, r_local, sigma_local, T)
        preco_call_adj = preco_call_just * markup_call

        call_justa_hist.append(preco_call_just)
        call_ajustada_hist.append(preco_call_adj)
        premio_call_pct.append(preco_call_adj / S0)

    ret_div = np.array(ret_div)
    premio_call_pct = np.array(premio_call_pct)
    cdi_periodos = np.array(cdi_periodos)
    sigma_local_hist = np.array(sigma_local_hist)

    # Retorno limitado pela call
    ret_limitado = np.minimum(ret_preco, ganho_max)

    # Retorno total da estratégia
    ret_total = ret_limitado + ret_div + premio_call_pct

    # Upside perdido por causa do strike
    upside_perdido = ret_preco - ret_limitado  # >= 0 quando há limitação

    # Regra: estrutura favorável se upside perdido <= prêmio da call
    deu_certo = (upside_perdido <= premio_call_pct).astype(int)

    bate_cdi = (ret_total > cdi_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_limitado": ret_limitado,
        "upside_perdido": upside_perdido,
        "ret_dividendos": ret_div,
        "premio_call_pct": premio_call_pct,
        "ret_total": ret_total,
        "cdi_periodo": cdi_periodos,
        "deu_certo": deu_certo,
        "bate_cdi": bate_cdi,
        "call_justa_hist": call_justa_hist,
        "call_ajustada_hist": call_ajustada_hist,
        "markup_call": markup_call,
        "sigma_local": sigma_local_hist,
        "cdi_detalhado": cdi_debug
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_cdi": bate_cdi.mean(),
        "preco_call_justa_hoje": preco_call_justa_hoje,
        "preco_call_cotada": preco_call_cotada,
        "markup_call": markup_call
    }

    return df, resumo, dividendos


# ============================================================
# GRÁFICOS
# ============================================================

def gerar_grafico_collar(df, ticker):
    dfp = df.copy()
    dfp["ret_ibov"] = gerar_ret_ibov(dfp)

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["data_inicio"], dfp["ret_op_com_div"], label=f"Collar – {ticker}")
    plt.plot(dfp["data_inicio"], dfp["ret_ibov"], label="IBOV")
    plt.axhline(0, color="black")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


def gerar_grafico_ap(df, ticker):
    dfp = df.copy()
    dfp["ret_ibov"] = gerar_ret_ibov(dfp)

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["data_inicio"], dfp["ret_ap_com_div"], label=f"AP – {ticker}")
    plt.plot(dfp["data_inicio"], dfp["ret_ibov"], label="IBOV")
    plt.axhline(0, color="black")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


def gerar_grafico_fin(df, ticker):
    dfp = df.copy()
    dfp["ret_ibov"] = gerar_ret_ibov(dfp)

    plt.figure(figsize=(12, 5))
    plt.plot(dfp["data_inicio"], dfp["ret_total"], label=f"Financiamento – {ticker}")
    plt.plot(dfp["data_inicio"], dfp["ret_ibov"], label="IBOV")
    plt.axhline(0, color="black")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# DASHBOARD STREAMLIT
# ============================================================

st.set_page_config(page_title="Backtest – Estruturas", layout="wide")

st.title("📈 Backtest – Collar, AP & Financiamento (CDI + spreads)")

st.markdown(
    "Backtest com **CDI (BACEN série 12)**, volatilidade histórica dinâmica e "
    "**spreads de PUT/CALL** para AP e Financiamento."
)

# Carregar CDI
try:
    cdi_df = carregar_cdi_com_fator()
except Exception:
    st.error("Erro ao carregar CDI (série 12) do BACEN.")
    st.stop()

tab_c, tab_ap, tab_fin = st.tabs(["📊 Collar", "🛡️ AP (PUT)", "💼 Financiamento (Covered Call)"])


# ------------------------------------------------------------
# COLLAR
# ------------------------------------------------------------
with tab_c:
    st.subheader("📊 Collar")

    ticker_c = st.text_input("Ticker:", "EZTC3.SA", key="t_c")
    prazo_du_c = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="p_c")
    ganho_c = st.number_input("Ganho Máx (%)", 0.0, 50.0, 8.0, key="g_c") / 100
    perda_c = st.number_input("Perda Máx (%)", 0.0, 50.0, 8.0, key="l_c") / 100

    if st.button("Rodar Collar"):
        precos, divs = carregar_preco_e_dividendos(ticker_c)
        resultado = backtest_collar(precos, divs, cdi_df, prazo_du_c, ganho_c, perda_c)

        if resultado:
            df_c, resumo_c, _ = resultado

            col1, col2 = st.columns(2)
            col1.metric("Estrutura Favorável (%)", f"{100*resumo_c['pct_deu_certo']:.1f}%")
            col2.metric("Bateu CDI (%)", f"{100*resumo_c['pct_bate_cdi']:.1f}%")

            st.subheader("Gráfico – Collar x IBOV")
            st.image(gerar_grafico_collar(df_c, ticker_c))

            st.subheader("Detalhamento")
            st.dataframe(df_c)


# ------------------------------------------------------------
# AP – ALOCAÇÃO PROTEGIDA (PUT)
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Alocação Protegida (PUT com spread)")

    ticker_ap = st.text_input("Ticker:", "EZTC3.SA", key="t_ap")
    prazo_du_ap = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="p_ap")
    perda_ap = st.number_input("Perda Máx Protegida (%)", 0.0, 50.0, 5.0, key="l_ap") / 100
    preco_put_cotada = st.number_input("Preço cotado da PUT hoje (R$):", 0.01, 50.0, 0.50, key="put_ap")

    if st.button("Rodar AP"):
        precos, divs = carregar_preco_e_dividendos(ticker_ap)
        resultado = backtest_ap(precos, divs, cdi_df, prazo_du_ap, perda_ap, preco_put_cotada)

        if resultado:
            df_ap, resumo_ap, _ = resultado

            c1, c2, c3 = st.columns(3)
            c1.metric("PUT justa hoje (BSL)", f"R$ {resumo_ap['preco_put_justa_hoje']:.4f}")
            c2.metric("PUT cotada", f"R$ {resumo_ap['preco_put_cotada']:.4f}")
            c3.metric("Spread PUT aplicado", f"{(resumo_ap['markup_put']-1)*100:.1f}%")

            c4, c5 = st.columns(2)
            c4.metric("Estrutura Favorável (%)", f"{100*resumo_ap['pct_deu_certo']:.1f}%")
            c5.metric("Bateu CDI (%)", f"{100*resumo_ap['pct_bate_cdi']:.1f}%")

            st.subheader("Gráfico – AP x IBOV")
            st.image(gerar_grafico_ap(df_ap, ticker_ap))

            st.subheader("Detalhamento")
            st.dataframe(df_ap)


# ------------------------------------------------------------
# FINANCIAMENTO / COVERED CALL
# ------------------------------------------------------------
with tab_fin:
    st.subheader("💼 Financiamento (Covered Call com spread)")

    ticker_f = st.text_input("Ticker:", "EZTC3.SA", key="t_fin")
    prazo_du_f = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="p_fin")
    ganho_f = st.number_input("Ganho Máx (%) – strike da CALL acima do spot", 0.0, 50.0, 8.0, key="g_fin") / 100
    preco_call_cotada = st.number_input("Preço cotado da CALL hoje (R$):", 0.01, 50.0, 0.50, key="call_fin")

    if st.button("Rodar Financiamento"):
        precos, divs = carregar_preco_e_dividendos(ticker_f)
        resultado = backtest_financiamento(precos, divs, cdi_df, prazo_du_f, ganho_f, preco_call_cotada)

        if resultado:
            df_fin, resumo_fin, _ = resultado

            c1, c2, c3 = st.columns(3)
            c1.metric("CALL justa hoje (BSL)", f"R$ {resumo_fin['preco_call_justa_hoje']:.4f}")
            c2.metric("CALL cotada", f"R$ {resumo_fin['preco_call_cotada']:.4f}")
            c3.metric("Spread CALL aplicado", f"{(resumo_fin['markup_call']-1)*100:.1f}%")

            c4, c5 = st.columns(2)
            c4.metric("Estrutura Favorável (%)", f"{100*resumo_fin['pct_deu_certo']:.1f}%")
            c5.metric("Bateu CDI (%)", f"{100*resumo_fin['pct_bate_cdi']:.1f}%")

            st.subheader("Gráfico – Financiamento x IBOV")
            st.image(gerar_grafico_fin(df_fin, ticker_f))

            st.subheader("Detalhamento")
            st.dataframe(df_fin)
