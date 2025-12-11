import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from math import log, sqrt, exp, erf
import requests
import datetime
import os

# ============================================================
# CONFIG GERAL – TEMA ESCURO + ROXO
# ============================================================

st.set_page_config(
    page_title="Backtest – Monte Bravo",
    page_icon="💜",
    layout="wide"
)

# CSS customizado
st.markdown("""
<style>
body {
    background-color: #0b0b10;
    color: #f2f2ff;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
header, .css-18e3th9, .css-1d391kg {
    background: #0b0b10;
}
h1, h2, h3, h4 {
    color: #e0d3ff !important;
}
.sidebar .sidebar-content {
    background-color: #111119 !important;
}
.css-1cypcdb {
    background-color: #111119 !important;
}
.metric-card {
    background-color: #151520;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid #272746;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOGO / HEADER MONTE BRAVO
# ============================================================

LOGO_PATH = "assets/monte_bravo_roxo.png"

col_logo, col_title = st.columns([1, 3])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
    else:
        st.markdown("### Monte Bravo")
        st.caption("Adicione sua logo em `assets/monte_bravo_roxo.png` para aparecer aqui.")

with col_title:
    st.markdown("## 💜 Backtest de Estruturas – Monte Bravo Premium")
    st.caption(
        "Ambiente institucional de simulação de **Collar** e **Alocação Protegida (AP)** "
        "com preços reais, dividendos, volatilidade histórica dinâmica e **Selic Meta (COPOM)**."
    )

DIAS_ANO = 252


# ============================================================
# SELIC META (COPOM) – SÉRIE 432 (ANUAL %)
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_selic_meta_com_fator():
    """
    Carrega SELIC META (série 432 do SGS), converte em:
    - valor: taxa anual (decimal)
    - fator_diario: taxa diária equivalente
    - série diária 'em degraus' entre reuniões do COPOM
    """
    hoje = datetime.date.today()
    data_final = hoje.strftime("%d/%m/%Y")
    data_inicial = (hoje - datetime.timedelta(days=3*365)).strftime("%d/%m/%Y")

    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/"
        f"dados?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float) / 100.0  # %a.a. → decimal
    df = df.set_index("data").sort_index()

    # Série diária em degraus (business days)
    diaria = df.resample("B").ffill()
    diaria["fator_diario"] = (1 + diaria["valor"]) ** (1 / DIAS_ANO) - 1
    return diaria  # colunas: valor (anual), fator_diario


def riskfree_periodo(meta_df: pd.DataFrame, ini, fim) -> float:
    """Retorno acumulado da Selic Meta entre ini e fim, usando fator_diario."""
    serie = meta_df.loc[ini:fim]["fator_diario"]
    if serie.empty:
        return 0.0
    return (1 + serie).prod() - 1


def obter_selic_meta_anual(meta_df: pd.DataFrame, data) -> float:
    """Taxa Selic Meta anual (decimal) vigente na data (último valor até a data)."""
    serie = meta_df["valor"].loc[:data]
    if serie.empty:
        return meta_df["valor"].iloc[0]
    return serie.iloc[-1]


# ============================================================
# DADOS DE PREÇO / DIVIDENDOS / IBOV
# ============================================================

def carregar_preco_e_dividendos(ticker: str):
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


def gerar_ret_ibov_periodo(df_datas: pd.DataFrame):
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
# BLACK-SCHOLES E VOL
# ============================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return max(K - S0 * exp(-r * T), 0.0)

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return max(price, 0.0)


def estimar_vol_anual(precos: pd.Series) -> float:
    if len(precos) < 2:
        return 1e-6
    log_ret = np.log(precos / precos.shift(1)).dropna()
    if log_ret.empty:
        return 1e-6
    sigma_diaria = log_ret.std()
    return max(sigma_diaria * np.sqrt(DIAS_ANO), 1e-6)


# ============================================================
# BACKTEST – COLLAR
# ============================================================

def backtest_collar(precos, dividendos, selic_meta_df, prazo_du, ganho_max, perda_max):
    datas = precos.index
    if len(datas) <= prazo_du:
        return None

    p0 = precos.values[:-prazo_du]
    p1 = precos.values[prazo_du:]
    ret_preco = p1 / p0 - 1

    ret_div = []
    selic_periodos = []
    selic_anual_meta = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        soma_div = 0.0
        if not dividendos.empty:
            soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma_div / p0[i])

        rf_per = riskfree_periodo(selic_meta_df, ini, fim)
        selic_periodos.append(rf_per)

        selic_anual_meta.append(obter_selic_meta_anual(selic_meta_df, ini))

    ret_div = np.array(ret_div)
    selic_periodos = np.array(selic_periodos)
    selic_anual_meta = np.array(selic_anual_meta)

    # Collar payoff
    ret_defesa = np.where(ret_preco < -perda_max, -perda_max - ret_preco, 0.0)
    limit_ganho = np.where(ret_preco > ganho_max, ret_preco - ganho_max, 0.0)

    ret_op_sem_div = np.where(
        (ret_defesa == 0) & (limit_ganho == 0),
        ret_preco,
        np.where(ret_defesa > 0, ret_preco + ret_defesa, ret_preco - limit_ganho),
    )

    deu_certo = ((ret_defesa > 0) | (limit_ganho == 0)).astype(int)

    ret_op_com_div = ret_op_sem_div + ret_div

    # Bate selic: retorno no período vs selic acumulada no período
    bate_selic = (ret_op_com_div > selic_periodos).astype(int)

    df_tmp = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:],
    })
    ret_ibov_periodo = gerar_ret_ibov_periodo(df_tmp)

    # Retornos anualizados por operação (para gráfico)
    ret_anual_estrut = (1 + ret_op_com_div) ** (DIAS_ANO / prazo_du) - 1
    ret_anual_ibov = (1 + ret_ibov_periodo) ** (DIAS_ANO / prazo_du) - 1

    df = pd.DataFrame({
        "data_inicio": df_tmp["data_inicio"],
        "data_fim": df_tmp["data_fim"],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_op_com_div": ret_op_com_div,
        "ret_ibov_periodo": ret_ibov_periodo,
        "ret_anual_estrut": ret_anual_estrut,
        "ret_anual_ibov": ret_anual_ibov,
        "selic_periodo": selic_periodos,
        "selic_meta_anual": selic_anual_meta,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
    })

    resumo = {
        "pct_deu_certo": float(deu_certo.mean()) if len(deu_certo) > 0 else 0.0,
        "pct_bate_selic": float(bate_selic.mean()) if len(bate_selic) > 0 else 0.0,
    }

    return df, resumo, dividendos


def gerar_grafico_collar(df, ticker: str) -> BytesIO:
    plt.figure(figsize=(12, 5))
    plt.plot(df["data_inicio"], df["ret_anual_estrut"], label=f"Collar – {ticker}", linewidth=2)
    plt.plot(df["data_inicio"], df["ret_anual_ibov"], label="IBOV (anualizado)", linewidth=2, alpha=0.8)
    plt.plot(df["data_inicio"], df["selic_meta_anual"], label="Selic Meta Anual (COPOM)", linewidth=2, linestyle="--", color="#bb86fc")

    plt.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    plt.title("Retornos Anualizados – Collar x IBOV x Selic Meta", fontsize=14)
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno / Taxa Anual")
    plt.grid(True, alpha=0.2, color="#444444")
    plt.legend()
    plt.gcf().patch.set_facecolor("#0b0b10")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0b0b10")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# BACKTEST – ALOCAÇÃO PROTEGIDA (AP)
# ============================================================

def backtest_ap(precos, dividendos, selic_meta_df, prazo_du, perda_max):
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
    sigmas = []
    selic_periodos = []
    selic_anual_meta = []

    for i in range(len(p0)):
        ini = datas[i]
        fim = datas[i + prazo_du]

        soma_div = 0.0
        if not dividendos.empty:
            soma_div = dividendos.loc[(dividendos.index >= ini) & (dividendos.index <= fim)].sum()
        ret_div.append(soma_div / p0[i])

        hist_pre = precos.loc[:ini].tail(DIAS_ANO)
        sigma_local = estimar_vol_anual(hist_pre)
        if sigma_local <= 0:
            sigma_local = sigma_global
        sigmas.append(sigma_local)

        r_ano = obter_selic_meta_anual(selic_meta_df, ini)

        S0 = p0[i]
        K = S0 * (1 - perda_max)
        T = prazo_du / DIAS_ANO

        put_price = black_scholes_put(S0, K, r_ano, sigma_local, T)
        preco_put_bsl.append(put_price)
        custo_put_pct.append(put_price / S0)

        rf_per = riskfree_periodo(selic_meta_df, ini, fim)
        selic_periodos.append(rf_per)
        selic_anual_meta.append(r_ano)

    ret_div = np.array(ret_div)
    preco_put_bsl = np.array(preco_put_bsl)
    custo_put_pct = np.array(custo_put_pct)
    sigmas = np.array(sigmas)
    selic_periodos = np.array(selic_periodos)
    selic_anual_meta = np.array(selic_anual_meta)

    ret_ap_com_div = ret_preco + ret_div - custo_put_pct

    hedge_acionado = (ret_preco <= -perda_max).astype(int)
    deu_certo = ((hedge_acionado == 1) | (ret_ap_com_div >= 0)).astype(int)
    bate_selic = (ret_ap_com_div > selic_periodos).astype(int)

    df_tmp = pd.DataFrame({
        "data_inicio": datas[:-prazo_du],
        "data_fim": datas[prazo_du:]
    })
    ret_ibov_periodo = gerar_ret_ibov_periodo(df_tmp)

    ret_anual_ap = (1 + ret_ap_com_div) ** (DIAS_ANO / prazo_du) - 1
    ret_anual_ibov = (1 + ret_ibov_periodo) ** (DIAS_ANO / prazo_du) - 1

    df = pd.DataFrame({
        "data_inicio": df_tmp["data_inicio"],
        "data_fim": df_tmp["data_fim"],
        "ret_preco": ret_preco,
        "ret_dividendos": ret_div,
        "ret_ap_com_div": ret_ap_com_div,
        "ret_ibov_periodo": ret_ibov_periodo,
        "ret_anual_ap": ret_anual_ap,
        "ret_anual_ibov": ret_anual_ibov,
        "preco_put_bsl": preco_put_bsl,
        "custo_put_pct": custo_put_pct,
        "sigma_local": sigmas,
        "selic_periodo": selic_periodos,
        "selic_meta_anual": selic_anual_meta,
        "hedge_acionado": hedge_acionado,
        "deu_certo": deu_certo,
        "bate_selic": bate_selic,
    })

    resumo = {
        "pct_deu_certo": float(deu_certo.mean()) if len(deu_certo) > 0 else 0.0,
        "pct_bate_selic": float(bate_selic.mean()) if len(bate_selic) > 0 else 0.0,
    }

    return df, resumo, dividendos


def gerar_grafico_ap(df, ticker: str) -> BytesIO:
    plt.figure(figsize=(12, 5))
    plt.plot(df["data_inicio"], df["ret_anual_ap"], label=f"AP – {ticker}", linewidth=2)
    plt.plot(df["data_inicio"], df["ret_anual_ibov"], label="IBOV (anualizado)", linewidth=2, alpha=0.8)
    plt.plot(df["data_inicio"], df["selic_meta_anual"], label="Selic Meta Anual (COPOM)", linewidth=2, linestyle="--", color="#bb86fc")

    plt.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    plt.title("Retornos Anualizados – AP x IBOV x Selic Meta", fontsize=14)
    plt.xlabel("Data de início da operação")
    plt.ylabel("Retorno / Taxa Anual")
    plt.grid(True, alpha=0.2, color="#444444")
    plt.legend()
    plt.gcf().patch.set_facecolor("#0b0b10")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0b0b10")
    buf.seek(0)
    plt.close()
    return buf


# ============================================================
# CARREGA SELIC META (UMA VEZ)
# ============================================================

try:
    selic_meta_df = carregar_selic_meta_com_fator()
except Exception as e:
    st.error(f"Erro ao carregar Selic Meta do BACEN: {e}")
    st.stop()


# ============================================================
# LAYOUT PRINCIPAL – TABS
# ============================================================

tab_collar, tab_ap = st.tabs(["📊 Collar", "🛡️ Alocação Protegida (AP)"])


# ------------------------------------------------------------
# COLLAR
# ------------------------------------------------------------
with tab_collar:
    st.subheader("📊 Estratégia Collar – Monte Bravo")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – Collar**")

    ticker_c = st.sidebar.text_input("Ticker – Collar", "EZTC3.SA", key="ticker_c")
    prazo_du_c = st.sidebar.number_input("Prazo (dias úteis)", 10, 252, 63, key="prazo_c")
    ganho_max_c = st.sidebar.number_input("Ganho Máximo (%)", 0.0, 50.0, 8.0, key="ganho_c") / 100
    perda_max_c = st.sidebar.number_input("Perda Máxima (%)", 0.0, 50.0, 8.0, key="perda_c") / 100

    rodar_c = st.sidebar.button("🚀 Rodar Collar", key="rodar_c")

    if rodar_c:
        precos_c, div_c = carregar_preco_e_dividendos(ticker_c)
        resultado_c = backtest_collar(precos_c, div_c, selic_meta_df, prazo_du_c, ganho_max_c, perda_max_c)

        if resultado_c is None:
            st.error("Histórico insuficiente para este prazo.")
        else:
            df_c, resumo_c, div_c = resultado_c

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Estrutura Favorável", f"{resumo_c['pct_deu_certo']*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Bateu Selic (período)", f"{resumo_c['pct_bate_selic']*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Gráfico – Retornos Anualizados vs Selic Meta")
            graf_c = gerar_grafico_collar(df_c, ticker_c)
            st.image(graf_c)

            st.markdown("### Detalhamento das Operações – Collar")
            st.dataframe(df_c)


# ------------------------------------------------------------
# AP – ALOCAÇÃO PROTEGIDA
# ------------------------------------------------------------
with tab_ap:
    st.subheader("🛡️ Estratégia Alocação Protegida – Monte Bravo")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Configurações – AP**")

    ticker_ap = st.sidebar.text_input("Ticker – AP", "EZTC3.SA", key="ticker_ap")
    prazo_du_ap = st.sidebar.number_input("Prazo (dias úteis) – AP", 10, 252, 63, key="prazo_ap")
    perda_max_ap = st.sidebar.number_input("Perda Máxima Protegida (%)", 0.0, 50.0, 5.0, key="perda_ap") / 100

    rodar_ap = st.sidebar.button("🛡️ Rodar AP", key="rodar_ap")

    if rodar_ap:
        precos_ap, div_ap = carregar_preco_e_dividendos(ticker_ap)
        resultado_ap = backtest_ap(precos_ap, div_ap, selic_meta_df, prazo_du_ap, perda_max_ap)

        if resultado_ap is None:
            st.error("Histórico insuficiente para este prazo.")
        else:
            df_ap, resumo_ap, div_ap = resultado_ap

            # Preço justo atual da put (para display)
            sigma_atual = estimar_vol_anual(precos_ap.tail(DIAS_ANO))
            S0_atual = precos_ap.iloc[-1]
            data_atual = precos_ap.index[-1]
            r_ano_atual = obter_selic_meta_anual(selic_meta_df, data_atual)
            K_atual = S0_atual * (1 - perda_max_ap)
            T_atual = prazo_du_ap / DIAS_ANO
            preco_put_hoje = black_scholes_put(S0_atual, K_atual, r_ano_atual, sigma_atual, T_atual)
            custo_put_pct_hoje = preco_put_hoje / S0_atual

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Estrutura Favorável", f"{resumo_ap['pct_deu_certo']*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Bateu Selic (período)", f"{resumo_ap['pct_bate_selic']*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("PUT Justa Hoje (R$)", f"{preco_put_hoje:.4f}")
                st.metric("Custo PUT (% do ativo)", f"{custo_put_pct_hoje*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### Gráfico – Retornos Anualizados vs Selic Meta")
            graf_ap = gerar_grafico_ap(df_ap, ticker_ap)
            st.image(graf_ap)

            st.markdown("### Detalhamento das Operações – AP")
            st.dataframe(df_ap)
