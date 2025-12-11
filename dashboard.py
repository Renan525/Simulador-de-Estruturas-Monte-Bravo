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
# SELIC HISTÓRICA – Banco Central
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_selic_com_fator():
    hoje = datetime.date.today()
    data_final = hoje.strftime("%d/%m/%Y")
    data_inicial = (hoje - datetime.timedelta(days=3*365)).strftime("%d/%m/%Y")

    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/"
        f"dados?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float) / 100.0      # TR = taxa anual (%)
    df = df.set_index("data").sort_index()

    # Fator diário equivalente
    df["fator_diario"] = (1 + df["valor"]) ** (1 / DIAS_ANO) - 1

    return df


def riskfree_periodo(selic_df: pd.DataFrame, ini, fim):
    """
    Retorna a Selic acumulada do período + a lista usada para debug.
    """

    # Range contínuo diário
    idx = pd.date_range(start=ini, end=fim, freq="D")

    # Forward-fill para garantir taxa válida em todos os dias do período
    serie = selic_df["fator_diario"].reindex(idx).ffill()

    # Valor acumulado
    acumulado = (1 + serie).prod() - 1

    return acumulado, serie


def obter_r_ano_selic(selic_df: pd.DataFrame, data) -> float:
    serie = selic_df["valor"].loc[:data]
    return serie.iloc[-1] if not serie.empty else selic_df["valor"].iloc[0]


# ============================================================
# BLACK-SCHOLES E VOL HISTÓRICA
# ============================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def estimar_vol_anual(precos: pd.Series):
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    return log_ret.std() * np.sqrt(252)


def black_scholes_put(S0, K, r, sigma, T):
    if T <= 0:
        return max(K - S0, 0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    preco = K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return max(preco, 0)


# ============================================================
# COLLAR
# ============================================================

def backtest_collar(precos, dividendos, selic_df, prazo_du, ganho_max, perda_max):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    selic_periodos = []
    selic_series_debug = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        # dividendos
        soma = 0.0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

        # Selic acumulada REAL + lista usada
        selic_acum, lista = riskfree_periodo(selic_df, ini, fim)
        selic_periodos.append(selic_acum)
        selic_series_debug.append(lista)

    ret_div = np.array(ret_div)
    selic_periodos = np.array(selic_periodos)

    # Payoff
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)
    ret_op_com_div = ret_op_sem_div + ret_div

    bate_selic = (ret_op_com_div > selic_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_sem_div": ret_op_sem_div,
        "ret_op_com_div": ret_op_com_div,
        "selic_periodo": selic_periodos,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
        "selic_detalhada": selic_series_debug,
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df, ticker):
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_op_com_div"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black")
    plt.title("Retornos Collar x IBOV")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# AP – Alocação Protegida
# ============================================================

def backtest_ap(precos, dividendos, selic_df, prazo_du, perda_max):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    sigma_global = estimar_vol_anual(precos)

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    preco_put_bsl = []
    custo_put_pct = []
    selic_periodos = []
    sigmas_usadas = []
    selic_debug = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        # dividendos
        soma = 0
        if not dividendos.empty:
            soma = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma / p0[i])

        # vol local
        hist_pre = precos.loc[:ini].tail(252)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigmas_usadas.append(sigma_local)

        # taxa anual Selic (precificação)
        r_local = obter_r_ano_selic(selic_df, ini)

        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / 252

        preco_put = black_scholes_put(S0, K, r_local, sigma_local, T)
        preco_put_bsl.append(preco_put)
        custo_put_pct.append(preco_put / S0)

        # Selic período
        selic_acum, lista = riskfree_periodo(selic_df, ini, fim)
        selic_periodos.append(selic_acum)
        selic_debug.append(lista)

    ret_div = np.array(ret_div)
    preco_put_bsl = np.array(preco_put_bsl)
    custo_put_pct = np.array(custo_put_pct)
    selic_periodos = np.array(selic_periodos)
    sigmas_usadas = np.array(sigmas_usadas)

    ret_ap_sem_div = ret_preco - custo_put_pct
    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)
    bate_selic = (ret_ap_com_div > selic_periodos).astype(int)

    df = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
        "preco_put_bsl": preco_put_bsl,
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "custo_put_pct": custo_put_pct,
        "ret_ap_sem_div": ret_ap_sem_div,
        "ret_ap_com_div": ret_ap_com_div,
        "selic_periodo": selic_periodos,
        "hedge_acionado": hedge_acionado,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
        "sigma_local": sigmas_usadas,
        "selic_detalhada": selic_debug,
    })

    resumo = {
        "pct_deu_certo": deu_certo.mean(),
        "pct_bate_selic": bate_selic.mean(),
    }

    return df, resumo, dividendos


def gerar_grafico_ap(df, ticker):
    df_plot = df.copy()
    df_plot["ret_ibov"] = gerar_ret_ibov(df_plot)

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["data_inicio"], df_plot["ret_ap_com_div"], label=f"AP – {ticker}", linewidth=2)
    plt.plot(df_plot["data_inicio"], df_plot["ret_ibov"], label="IBOV", linewidth=2, alpha=0.8)
    plt.axhline(0, color="black")
    plt.title("Retornos AP x IBOV")
    plt.grid(True, alpha=0.3)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# DASHBOARD
# ============================================================

st.set_page_config(page_title="Backtest – Estruturas", layout="wide")

st.title("📈 Backtest – Collar & Alocação Protegida")


st.markdown(
    """
    **Agora com Selic acumulada REAL por período**,  
    usando série oficial do Banco Central com interpolação diária.
    """
)

try:
    selic_df = carregar_selic_com_fator()
except:
    st.error("Erro ao carregar selic do Banco Central.")
    st.stop()

tab_c, tab_ap = st.tabs(["📊 Collar", "🛡️ AP"])


# ============================================================
# COLLAR
# ============================================================

with tab_c:
    st.subheader("📊 Collar")

    ticker = st.text_input("Ticker:", "EZTC3.SA", key="c_ticker")
    prazo = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="c_du")
    ganho = st.number_input("Ganho Máx (%)", 0.0, 30.0, 8.0, key="c_gain") / 100
    perda = st.number_input("Perda Máx (%)", 0.0, 30.0, 8.0, key="c_loss") / 100

    if st.button("Rodar Collar"):
        precos, divs = carregar_preco_e_dividendos(ticker)
        ret = backtest_collar(precos, divs, selic_df, prazo, ganho, perda)

        if ret is None:
            st.error("Histórico insuficiente.")
        else:
            df, resumo, _ = ret

            c1, c2 = st.columns(2)
            c1.metric("Estrutura Favorável (%)", f"{100*resumo['pct_deu_certo']:.1f}%")
            c2.metric("Bateu Selic (%)", f"{100*resumo['pct_bate_selic']:.1f}%")

            st.subheader("Gráfico")
            st.image(gerar_grafico_collar(df, ticker))

            st.subheader("Detalhamento (com Selic detalhada)")
            st.dataframe(df)


# ============================================================
# AP
# ============================================================

with tab_ap:
    st.subheader("🛡️ Alocação Protegida (AP)")

    ticker = st.text_input("Ticker:", "EZTC3.SA", key="ap_ticker")
    prazo = st.number_input("Prazo (dias úteis)", 10, 252, 63, key="ap_du")
    perda = st.number_input("Perda Máxima Protegida (%)", 0.0, 30.0, 5.0, key="ap_loss") / 100

    if st.button("Rodar AP"):
        precos, divs = carregar_preco_e_dividendos(ticker)
        ret = backtest_ap(precos, divs, selic_df, prazo, perda)

        if ret is None:
            st.error("Histórico insuficiente.")
        else:
            df, resumo, _ = ret

            # Preço da put hoje
            sigma_atual = estimar_vol_anual(precos.tail(252))
            S0_atual = precos.iloc[-1]
            K = S0_atual * (1 - perda)
            T = prazo / 252
            r_atual = obter_r_ano_selic(selic_df, precos.index[-1])

            preco_put_hoje = black_scholes_put(S0_atual, K, r_atual, sigma_atual, T)
            pct_hoje = preco_put_hoje / S0_atual

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estrutura Favorável (%)", f"{100*resumo['pct_deu_certo']:.1f}%")
            c2.metric("Bateu Selic (%)", f"{100*resumo['pct_bate_selic']:.1f}%")
            c3.metric("Put justa hoje (R$)", f"R$ {preco_put_hoje:.4f}")
            c4.metric("Put (% do ativo)", f"{100*pct_hoje:.2f}%")

            st.subheader("Gráfico")
            st.image(gerar_grafico_ap(df, ticker))

            st.subheader("Detalhamento (com Selic detalhada)")
            st.dataframe(df)

