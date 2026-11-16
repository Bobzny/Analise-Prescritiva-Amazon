from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    stats = None
    HAS_SCIPY = False

st.set_page_config(
    page_title="Análise Prescritiva • Amazon Delivery & Gestão de Estoque",
    layout="wide",
)

PASTA_APP = Path(__file__).resolve().parent

# Mapas de tradução reversa (PT-BR -> Inglês original do Kaggle)
PT_PARA_EN_PRODUTO = {
    "Eletrônicos": "Electronics",
    "Livros": "Books",
    "Joias": "Jewelry",
    "Brinquedos": "Toys",
    "Cuidados com a Pele": "Skincare",
    "Lanches": "Snacks",
    "Ar Livre e Camping": "Outdoors",
    "Vestuário": "Apparel",
    "Esportes": "Sports",
    "Mercado (Perecíveis)": "Grocery",
    "Pet Shop": "Pet Supplies",
    "Casa e Decoração": "Home",
    "Cosméticos": "Cosmetics",
    "Cozinha": "Kitchen",
    "Roupas": "Clothing",
    "Calçados": "Shoes",
}
PT_PARA_EN_CLIMA = {
    "Ensolarado": "Sunny",
    "Tempestade": "Stormy",
    "Nublado": "Cloudy",
    "Neblina": "Fog",
    "Ventania": "Windy",
    "Tempestade de Areia": "Sandstorms",
    "Não Informado": "Unknown",
}
PT_PARA_EN_TRANSITO = {
    "Livre": "Low",
    "Médio": "Medium",
    "Alto": "High",
    "Engarrafamento": "Jam",
    "Não Informado": "Unknown",
}
PT_PARA_EN_VEICULO = {
    "Moto": "motorcycle",
    "Scooter": "scooter",
    "Van": "van",
    "Bicicleta": "bicycle",
}
PT_PARA_EN_REGIAO = {
    "Urbana": "Urban",
    "Metropolitana": "Metropolitan",
    "Semiurbana": "Semi-Urban",
    "Outra": "Other",
}

DICIONARIO_VARIAVEIS = [
    {"Variável PT-BR": "id_pedido", "Original (Inglês)": "Order_ID", "Descrição": "Identificador único da entrega"},
    {"Variável PT-BR": "data_pedido", "Original (Inglês)": "Order_Date", "Descrição": "Data do pedido (44 dias de operação)"},
    {"Variável PT-BR": "hora_pedido", "Original (Inglês)": "Order_Time", "Descrição": "Hora em que o pedido foi realizado"},
    {"Variável PT-BR": "hora_coleta", "Original (Inglês)": "Pickup_Time", "Descrição": "Hora em que o entregador coletou na loja"},
    {"Variável PT-BR": "produto", "Original (Inglês)": "Category", "Descrição": "Categoria do produto (16 categorias)"},
    {"Variável PT-BR": "regiao", "Original (Inglês)": "Area", "Descrição": "Tipo de região (Urbana, Metropolitana, etc.)"},
    {"Variável PT-BR": "clima", "Original (Inglês)": "Weather", "Descrição": "Condição climática na entrega"},
    {"Variável PT-BR": "transito", "Original (Inglês)": "Traffic", "Descrição": "Nível de trânsito na entrega"},
    {"Variável PT-BR": "veiculo", "Original (Inglês)": "Vehicle", "Descrição": "Veículo utilizado pelo entregador"},
    {"Variável PT-BR": "idade_entregador", "Original (Inglês)": "Agent_Age", "Descrição": "Idade do entregador"},
    {"Variável PT-BR": "nota_entregador", "Original (Inglês)": "Agent_Rating", "Descrição": "Avaliação do entregador"},
    {"Variável PT-BR": "distancia_km", "Original (Inglês)": "Distance_km", "Descrição": "Distância Loja–Cliente (Haversine pós-correção GPS)"},
    {"Variável PT-BR": "tempo_entrega_min", "Original (Inglês)": "Delivery_Time", "Descrição": "Tempo total da entrega (em minutos)"},
    {"Variável PT-BR": "tempo_entrega_horas", "Original (Inglês)": "Delivery_Time_Hours", "Descrição": "Tempo de entrega convertido para horas"},
    {"Variável PT-BR": "tempo_entrega_dias", "Original (Inglês)": "Delivery_Time_Days", "Descrição": "Tempo de entrega convertido para dias"},
]

NIVEIS_SERVICO_Z = {
    "90% (z=1,28)": 1.28,
    "95% (z=1,65)": 1.65,
    "97,5% (z=1,96)": 1.96,
    "99% (z=2,33)": 2.33,
}


@st.cache_data
def ler_csv_disco(caminho_str: str) -> pd.DataFrame:
    """Lê um CSV do disco tentando UTF-8 com BOM e fallback para Latin-1."""
    try:
        return pd.read_csv(caminho_str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(caminho_str, encoding="latin-1")


def ler_csv_upload(arquivo) -> pd.DataFrame:
    """Lê um arquivo CSV enviado pelo file_uploader tratando codificação."""
    try:
        arquivo.seek(0)
        return pd.read_csv(arquivo, encoding="utf-8-sig")
    except UnicodeDecodeError:
        arquivo.seek(0)
        return pd.read_csv(arquivo, encoding="latin-1")


def carregar_tabela(nome_arquivo: str, rotulo: str) -> pd.DataFrame | None:
    """Carrega o CSV da pasta do aplicativo ou solicita upload pela barra lateral."""
    caminho = PASTA_APP / nome_arquivo
    if caminho.exists():
        return ler_csv_disco(str(caminho))
    arquivo = st.sidebar.file_uploader(
        f"Envie {rotulo} ({nome_arquivo})", type=["csv"], key=f"up_{nome_arquivo}"
    )
    return ler_csv_upload(arquivo) if arquivo is not None else None


def validar_e_limpar_bases(
    pedidos: pd.DataFrame, estoque: pd.DataFrame, parametros: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Valida colunas obrigatórias e garante tipagem numérica e temporal correta."""
    req_pedidos = [
        "data_pedido", "produto", "regiao", "clima", "transito",
        "veiculo", "distancia_km", "tempo_entrega_min",
    ]
    req_estoque = ["data_pedido", "produto", "demanda_diaria", "distancia_media_km"]
    req_param = ["produto", "demanda_media_diaria", "demanda_desvio_diario"]

    for df_alvo, cols, nome in [
        (pedidos, req_pedidos, "amazon_pedidos_limpo.csv"),
        (estoque, req_estoque, "amazon_estoque_diario.csv"),
        (parametros, req_param, "amazon_parametros_produtos.csv"),
    ]:
        ausentes = [c for c in cols if c not in df_alvo.columns]
        if ausentes:
            st.error(f"No arquivo `{nome}`, faltam as colunas: {', '.join(ausentes)}")
            st.stop()

    pedidos = pedidos.copy()
    estoque = estoque.copy()
    parametros = parametros.copy()

    pedidos["data_pedido"] = pd.to_datetime(pedidos["data_pedido"], errors="coerce")
    pedidos["distancia_km"] = pd.to_numeric(pedidos["distancia_km"], errors="coerce")
    pedidos["tempo_entrega_min"] = pd.to_numeric(pedidos["tempo_entrega_min"], errors="coerce")

    estoque["data_pedido"] = pd.to_datetime(estoque["data_pedido"], errors="coerce")
    estoque["demanda_diaria"] = pd.to_numeric(estoque["demanda_diaria"], errors="coerce")
    estoque["distancia_media_km"] = pd.to_numeric(estoque["distancia_media_km"], errors="coerce")

    parametros["demanda_media_diaria"] = pd.to_numeric(parametros["demanda_media_diaria"], errors="coerce")
    parametros["demanda_desvio_diario"] = pd.to_numeric(parametros["demanda_desvio_diario"], errors="coerce")

    pedidos = pedidos.dropna(subset=req_pedidos).reset_index(drop=True)
    estoque = estoque.dropna(subset=req_estoque).reset_index(drop=True)
    parametros = parametros.dropna(subset=req_param).reset_index(drop=True)

    return pedidos, estoque, parametros


def aplicar_idioma(
    pedidos: pd.DataFrame, estoque: pd.DataFrame, parametros: pd.DataFrame, em_portugues: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna cópias dos DataFrames no idioma selecionado (PT-BR ou Inglês original)."""
    if em_portugues:
        return pedidos, estoque, parametros

    ped_en = pedidos.copy()
    est_en = estoque.copy()
    par_en = parametros.copy()

    ped_en["produto"] = ped_en["produto"].replace(PT_PARA_EN_PRODUTO)
    ped_en["clima"] = ped_en["clima"].replace(PT_PARA_EN_CLIMA)
    ped_en["transito"] = ped_en["transito"].replace(PT_PARA_EN_TRANSITO)
    ped_en["veiculo"] = ped_en["veiculo"].replace(PT_PARA_EN_VEICULO)
    ped_en["regiao"] = ped_en["regiao"].replace(PT_PARA_EN_REGIAO)
    ped_en = ped_en.rename(columns={
        "id_pedido": "Order_ID", "data_pedido": "Order_Date", "hora_pedido": "Order_Time",
        "hora_coleta": "Pickup_Time", "produto": "Category", "regiao": "Area",
        "clima": "Weather", "transito": "Traffic", "veiculo": "Vehicle",
        "idade_entregador": "Agent_Age", "nota_entregador": "Agent_Rating",
        "distancia_km": "Distance_km", "tempo_entrega_min": "Delivery_Time",
        "tempo_entrega_horas": "Delivery_Time_Hours", "tempo_entrega_dias": "Delivery_Time_Days",
    })

    est_en["produto"] = est_en["produto"].replace(PT_PARA_EN_PRODUTO)
    est_en = est_en.rename(columns={
        "data_pedido": "Order_Date", "produto": "Category", "demanda_diaria": "Daily_Demand",
        "tempo_medio_horas": "Avg_Delivery_Hours", "tempo_desvio_horas": "Std_Delivery_Hours",
        "distancia_media_km": "Avg_Distance_km",
    })

    par_en["produto"] = par_en["produto"].replace(PT_PARA_EN_PRODUTO)
    par_en = par_en.rename(columns={
        "produto": "Category", "dias_analisados": "Days_Analyzed", "total_pedidos": "Total_Orders",
        "demanda_media_diaria": "Avg_Daily_Demand", "demanda_desvio_diario": "Std_Daily_Demand",
        "tempo_medio_horas": "Avg_Delivery_Hours", "tempo_desvio_horas": "Std_Delivery_Hours",
        "distancia_media_km": "Avg_Distance_km",
    })

    return ped_en, est_en, par_en


def ajustar_regressao(x, y) -> dict[str, float]:
    """Ajusta regressão linear simples por mínimos quadrados com fallback em NumPy caso SciPy não esteja instalado."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(x_arr) < 2 or np.std(x_arr) == 0:
        media_y = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.0
        return {"a": 0.0, "b": media_y, "r2": 0.0, "p": 1.0, "se": 0.0}
    if HAS_SCIPY:
        res = stats.linregress(x_arr, y_arr)
        return {
            "a": float(res.slope),
            "b": float(res.intercept),
            "r2": float(res.rvalue ** 2),
            "p": float(res.pvalue),
            "se": float(res.stderr) if res.stderr is not None else 0.0,
        }

    dx = x_arr - float(np.mean(x_arr))
    dy = y_arr - float(np.mean(y_arr))
    sxx = float(np.dot(dx, dx))
    syy = float(np.dot(dy, dy))
    sxy = float(np.dot(dx, dy))
    a = sxy / sxx
    b = float(np.mean(y_arr)) - a * float(np.mean(x_arr))
    r2 = min(max((sxy * sxy) / (sxx * syy), 0.0), 1.0) if syy > 0 else 0.0
    n = len(x_arr)
    if n > 2:
        resid = y_arr - (a * x_arr + b)
        se = math.sqrt(max(float(np.dot(resid, resid)) / (n - 2), 0.0) / sxx)
        p = math.erfc(abs(a / se) / math.sqrt(2.0)) if se > 0 else (0.0 if a != 0 else 1.0)
    else:
        se, p = 0.0, 1.0
    return {"a": float(a), "b": float(b), "r2": float(r2), "p": float(p), "se": float(se)}


def classificar_tendencia(inclinacao: float) -> str:
    """Classifica inclinação da reta de demanda diária."""
    if inclinacao > 0.05:
        return "Crescente"
    if inclinacao < -0.05:
        return "Decrescente"
    return "Estável"


def formatar_p_valor(p: float) -> str:
    """Formata p-valor em notação científica quando muito pequeno."""
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def calcular_desempenho_ponderado(matriz: pd.DataFrame, freq_cenario: pd.Series) -> pd.DataFrame:
    """Calcula o tempo esperado ponderado E[T_v] tratando eventuais células ausentes."""
    desempenho = matriz.copy()
    pesos = freq_cenario.reindex(desempenho.columns).fillna(0.0).values.astype(float)
    valores_ponderados = []
    for _, linha in desempenho.iterrows():
        arr = linha.values.astype(float)
        valido = ~np.isnan(arr) & (pesos > 0)
        if valido.any() and pesos[valido].sum() > 0:
            valores_ponderados.append(float((arr[valido] * pesos[valido]).sum() / pesos[valido].sum()))
        else:
            valores_ponderados.append(np.nan)
    desempenho["Tempo médio ponderado"] = valores_ponderados
    return desempenho


def simular_estoque_44_dias(
    df_diario_prod: pd.DataFrame,
    col_data: str,
    col_demanda: str,
    ponto_de_pedido: int,
    lote_reposicao: int,
) -> pd.DataFrame:
    """Executa simulação iterativa diária do nível de estoque ao longo dos dias observados."""
    estoque_atual = max(ponto_de_pedido * 2, lote_reposicao)
    historico = []
    for _, linha in df_diario_prod.iterrows():
        vendas_hoje = int(linha[col_demanda])
        estoque_atual -= vendas_hoje
        pediu_reposicao = "Não"
        if estoque_atual <= ponto_de_pedido:
            estoque_atual += lote_reposicao
            pediu_reposicao = f"Sim (+{lote_reposicao} un)"
        historico.append({
            "dia_numero": int(linha["dia_operacao"]),
            "data": linha[col_data],
            "demanda_real_dia": vendas_hoje,
            "demanda_regressao_linear": round(float(linha["demanda_tendencia"]), 2),
            "estoque_fim_do_dia": max(0, int(estoque_atual)),
            "acionou_reposicao": pediu_reposicao,
        })
    return pd.DataFrame(historico)


def renderizar_bloco_sensibilidade(
    df_base: pd.DataFrame,
    col_grupo: str,
    col_tempo: str,
    titulo_grupo: str,
    valores_baseline: list[str],
    rotulo_baseline: str,
    escala_cor: str,
) -> None:
    """Renderiza tabela, gráfico de barras com desvio e boxplot para análise de sensibilidade."""
    sens = (
        df_base.groupby(col_grupo)[col_tempo]
        .agg(Quantidade="count", Tempo_medio="mean", Mediana="median", Desvio="std")
        .reset_index()
        .sort_values("Tempo_medio")
    )
    sens["Desvio"] = sens["Desvio"].fillna(0.0)

    mask_ref = sens[col_grupo].astype(str).str.strip().str.casefold().isin(valores_baseline)
    if mask_ref.any():
        base_tempo = float(sens.loc[mask_ref, "Tempo_medio"].iloc[0])
        if base_tempo > 0:
            sens[f"Δ vs. {rotulo_baseline} (min)"] = sens["Tempo_medio"] - base_tempo
            sens[f"Δ vs. {rotulo_baseline} (%)"] = (sens["Tempo_medio"] / base_tempo - 1.0) * 100.0

    col_g, col_b = st.columns(2)
    with col_g:
        fig_bar = px.bar(
            sens,
            x=col_grupo,
            y="Tempo_medio",
            text_auto=".1f",
            error_y="Desvio",
            color="Tempo_medio",
            color_continuous_scale=escala_cor,
            title=f"Tempo médio por {titulo_grupo} (barras de erro = ±1σ)",
            labels={col_grupo: titulo_grupo, "Tempo_medio": "Tempo médio (min)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_box = px.box(
            df_base,
            x=col_grupo,
            y=col_tempo,
            color=col_grupo,
            title=f"Distribuição do tempo de entrega por {titulo_grupo}",
            labels={col_grupo: titulo_grupo, col_tempo: "Tempo (min)"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.dataframe(sens.round(2), use_container_width=True, hide_index=True)


# =========================================================
# BARRA LATERAL — IDIOMA, CARREGAMENTO E PARÂMETROS
# =========================================================
st.sidebar.title("Configurações e Filtros")
idioma = st.sidebar.radio(
    "Idioma das Variáveis e Tabelas:",
    options=["Português (PT-BR)", "Inglês (Original)"],
    index=0,
)
usar_pt = idioma.startswith("Português")

with st.sidebar.expander("Fonte de Dados (CSVs)", expanded=False):
    st.caption("CSVs na mesma pasta do `app.py` carregam automaticamente; caso contrário, envie abaixo.")
    df_estoque_raw = carregar_tabela("amazon_estoque_diario.csv", "a tabela diária")
    df_param_raw = carregar_tabela("amazon_parametros_produtos.csv", "os parâmetros dos produtos")
    df_pedidos_raw = carregar_tabela("amazon_pedidos_limpo.csv", "os pedidos tratados")

faltando = [
    nome for nome, tabela in [
        ("amazon_estoque_diario.csv", df_estoque_raw),
        ("amazon_parametros_produtos.csv", df_param_raw),
        ("amazon_pedidos_limpo.csv", df_pedidos_raw),
    ]
    if tabela is None
]
if faltando:
    st.warning(
        "Coloque os três CSVs gerados por `limpeza.py` na pasta do `app.py` "
        "ou faça o upload na barra lateral (seção Fonte de Dados)."
    )
    for nome in faltando:
        st.write(f"- `{nome}`")
    st.stop()

df_pedidos_pt, df_estoque_pt, df_param_pt = validar_e_limpar_bases(
    df_pedidos_raw, df_estoque_raw, df_param_raw
)
df_pedidos, df_estoque, df_parametros = aplicar_idioma(
    df_pedidos_pt, df_estoque_pt, df_param_pt, usar_pt
)

# Nomes de colunas ativos conforme idioma escolhido
COL_PRODUTO = "produto" if usar_pt else "Category"
COL_CLIMA = "clima" if usar_pt else "Weather"
COL_TRANSITO = "transito" if usar_pt else "Traffic"
COL_VEICULO = "veiculo" if usar_pt else "Vehicle"
COL_REGIAO = "regiao" if usar_pt else "Area"
COL_TEMPO_MIN = "tempo_entrega_min" if usar_pt else "Delivery_Time"
COL_DISTANCIA = "distancia_km" if usar_pt else "Distance_km"
COL_ESTOQUE_DATA = "data_pedido" if usar_pt else "Order_Date"
COL_ESTOQUE_DEMANDA = "demanda_diaria" if usar_pt else "Daily_Demand"
COL_ESTOQUE_DIST = "distancia_media_km" if usar_pt else "Avg_Distance_km"
COL_PARAM_MEDIA = "demanda_media_diaria" if usar_pt else "Avg_Daily_Demand"
COL_PARAM_DESVIO = "demanda_desvio_diario" if usar_pt else "Std_Daily_Demand"

# Colunas auxiliares normalizadas para comparações seguras em ambos os idiomas
df_pedidos["_clima_norm"] = df_pedidos[COL_CLIMA].astype(str).str.strip().str.casefold()
df_pedidos["_transito_norm"] = df_pedidos[COL_TRANSITO].astype(str).str.strip().str.casefold()

# Mapeamento sequencial de dia de operação (1..N) respeitando eventuais lacunas no calendário
datas_unicas = sorted(df_estoque[COL_ESTOQUE_DATA].unique())
mapa_dia_op = {d: i + 1 for i, d in enumerate(datas_unicas)}
df_estoque["dia_operacao"] = df_estoque[COL_ESTOQUE_DATA].map(mapa_dia_op)
data_inicial = df_estoque[COL_ESTOQUE_DATA].min()
data_final = df_estoque[COL_ESTOQUE_DATA].max()
n_cal = int((data_final - data_inicial).days) + 1
n_dado = int(df_estoque[COL_ESTOQUE_DATA].nunique())

# Filtros de Produto e Cenário
st.sidebar.divider()
st.sidebar.subheader("Cenário e Parâmetros da Simulação")

OPCAO_TODOS = "Todos" if usar_pt else "All"
lista_produtos = sorted(df_pedidos[COL_PRODUTO].unique())
padrao_prod = "Eletrônicos" if usar_pt else "Electronics"
produto_sel = st.sidebar.selectbox(
    f"Categoria de Produto ({COL_PRODUTO}):",
    options=lista_produtos,
    index=lista_produtos.index(padrao_prod) if padrao_prod in lista_produtos else 0,
)

lista_climas = sorted(df_pedidos[COL_CLIMA].unique())
clima_sel = st.sidebar.selectbox(
    f"Clima ({COL_CLIMA}):",
    options=[OPCAO_TODOS] + lista_climas,
)

lista_transitos = sorted(df_pedidos[COL_TRANSITO].unique())
transito_sel = st.sidebar.selectbox(
    f"Trânsito ({COL_TRANSITO}):",
    options=[OPCAO_TODOS] + lista_transitos,
)

prazo_L = st.sidebar.slider("Tempo de Reposição (dias):", min_value=1, max_value=30, value=3)

metodo_es = st.sidebar.radio(
    "Modelo de Cálculo do Ponto de Reposição:",
    options=[
        "Reserva de Segurança (%)",
        "Nível de Serviço (Estatístico)",
    ],
    index=0,
)
if metodo_es == "Nível de Serviço (Estatístico)":
    nivel_servico_sel = st.sidebar.selectbox(
        "Nível de Serviço:",
        options=list(NIVEIS_SERVICO_Z.keys()),
        index=1,
    )
    margem_seguranca_pct = 20
else:
    margem_seguranca_pct = st.sidebar.slider(
        "Reserva de Segurança (%):",
        min_value=0, max_value=100, value=20, step=5,
    )
    nivel_servico_sel = "95% (z=1,65)"

z_score = NIVEIS_SERVICO_Z[nivel_servico_sel]

# =========================================================
# FILTRAGEM DE DADOS E CÁLCULOS PRESCRITIVOS
# =========================================================
df_prod = df_pedidos[df_pedidos[COL_PRODUTO] == produto_sel].copy()
df_cenario = df_prod.copy()
df_cenario_global = df_pedidos.copy()

if clima_sel != OPCAO_TODOS:
    df_cenario = df_cenario[df_cenario[COL_CLIMA] == clima_sel]
    df_cenario_global = df_cenario_global[df_cenario_global[COL_CLIMA] == clima_sel]
if transito_sel != OPCAO_TODOS:
    df_cenario = df_cenario[df_cenario[COL_TRANSITO] == transito_sel]
    df_cenario_global = df_cenario_global[df_cenario_global[COL_TRANSITO] == transito_sel]

aviso_cenario_vazio = False
if df_cenario.empty:
    aviso_cenario_vazio = True
    df_cenario = df_prod.copy()
if df_cenario_global.empty:
    df_cenario_global = df_pedidos.copy()

df_diario_prod = (
    df_estoque[df_estoque[COL_PRODUTO] == produto_sel]
    .sort_values("dia_operacao")
    .reset_index(drop=True)
)

total_pedidos_prod = len(df_prod)
media_pedidos_dia = float(df_diario_prod[COL_ESTOQUE_DEMANDA].mean())
desvio_pedidos_dia = float(df_diario_prod[COL_ESTOQUE_DEMANDA].std(ddof=1)) if len(df_diario_prod) > 1 else 0.0

# Regressão linear da demanda diária do produto selecionado
reg_dem = ajustar_regressao(df_diario_prod["dia_operacao"], df_diario_prod[COL_ESTOQUE_DEMANDA])
coef_a_dem, coef_b_dem, r2_dem, p_dem, se_dem = (
    reg_dem["a"], reg_dem["b"], reg_dem["r2"], reg_dem["p"], reg_dem["se"]
)
tendencia_dem = classificar_tendencia(coef_a_dem)
df_diario_prod["demanda_tendencia"] = (coef_a_dem * df_diario_prod["dia_operacao"] + coef_b_dem).round(2)

# Baseline (Livre + Ensolarado) e fator de atraso para Prazo Efetivo (L_ef)
mask_baseline = (
    df_pedidos["_transito_norm"].isin(["livre", "low"])
    & df_pedidos["_clima_norm"].isin(["ensolarado", "sunny"])
)
baseline_tempo_min = (
    float(df_pedidos.loc[mask_baseline, COL_TEMPO_MIN].mean())
    if mask_baseline.any()
    else float(df_pedidos[COL_TEMPO_MIN].mean())
)
cenario_tempo_global_min = float(df_cenario_global[COL_TEMPO_MIN].mean())
fator_atraso_pct = (
    max(cenario_tempo_global_min / baseline_tempo_min - 1.0, 0.0)
    if baseline_tempo_min > 0
    else 0.0
)
L_efetivo = prazo_L * (1.0 + fator_atraso_pct)

# Tempos do produto no cenário vs média do produto
tempo_medio_geral_min = float(df_prod[COL_TEMPO_MIN].mean())
tempo_medio_cenario_min = float(df_cenario[COL_TEMPO_MIN].mean())
diferenca_tempo_min = tempo_medio_cenario_min - tempo_medio_geral_min
fator_atraso_rel_prod = (tempo_medio_cenario_min / tempo_medio_geral_min) if tempo_medio_geral_min > 0 else 1.0

# Cálculo do Ponto de Reposição nos dois modelos (Estatístico e Percentual)
demanda_durante_prazo_ef = media_pedidos_dia * L_efetivo
es_estatistico = z_score * desvio_pedidos_dia * math.sqrt(L_efetivo)
pr_estatistico = math.ceil(demanda_durante_prazo_ef + es_estatistico)

estoque_basico_pct = media_pedidos_dia * prazo_L
es_percentual = estoque_basico_pct * (margem_seguranca_pct / 100.0) * fator_atraso_rel_prod
pr_percentual = math.ceil(estoque_basico_pct + es_percentual)

usar_estatistico = metodo_es.startswith("Nível de Serviço")
ponto_de_pedido = pr_estatistico if usar_estatistico else pr_percentual
es_ativo = es_estatistico if usar_estatistico else es_percentual
metodo_label_curto = f"z={z_score:.2f}".replace(".", ",") if usar_estatistico else f"{margem_seguranca_pct}%"

lote_reposicao = max(1, round(media_pedidos_dia * prazo_L * 1.5))
df_sim_real = simular_estoque_44_dias(
    df_diario_prod, COL_ESTOQUE_DATA, COL_ESTOQUE_DEMANDA, ponto_de_pedido, lote_reposicao
)
total_reposicoes_sim = int((df_sim_real["acionou_reposicao"] != "Não").sum())
dias_ruptura_sim = int((df_sim_real["estoque_fim_do_dia"] == 0).sum())
estoque_medio_sim = float(df_sim_real["estoque_fim_do_dia"].mean())
estoque_min_sim = int(df_sim_real["estoque_fim_do_dia"].min())

# Ranking de veículos no cenário filtrado do produto
ranking_veiculos_cenario = (
    df_cenario.groupby(COL_VEICULO)[COL_TEMPO_MIN]
    .agg(tempo_medio_min="mean", quantidade_entregas="count")
    .reset_index()
    .sort_values("tempo_medio_min")
)
melhor_veiculo_cenario = ranking_veiculos_cenario.iloc[0][COL_VEICULO]
melhor_tempo_cenario = float(ranking_veiculos_cenario.iloc[0]["tempo_medio_min"])

# =========================================================
# CABEÇALHO E KPIs PRINCIPAIS
# =========================================================
st.title("Estudo de Caso: Análise de Dados Prescritiva (Amazon Delivery Dataset)")
st.markdown(
    f"Painel interativo integrando **Etapa 1 (Tratamento e Limpeza dos Dados)**, "
    f"**Etapa 2 (Modelagem Matemática, Regressão Linear, Sensibilidade e Otimização de Frota)** e "
    f"**Etapa 3 (Ponto de Reposição e Simulação de Estoque)**. "
    f"Idioma ativo: **{idioma}** | Categoria de Produto: **{produto_sel}**."
)
st.caption(
    f"Base global: {len(df_pedidos):,} pedidos | {df_parametros.iloc[:, 0].nunique()} categorias | "
    f"{df_estoque[COL_ESTOQUE_DATA].nunique()} dias observados | "
    f"Baseline (Livre + Ensolarado): {baseline_tempo_min:.1f} min".replace(",", ".")
)

if aviso_cenario_vazio:
    st.warning(
        f"Não há entregas para `{produto_sel}` na combinação `{clima_sel}` + `{transito_sel}`. "
        "Exibindo todos os dados do produto no cenário."
    )

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "1. Demanda Média Diária",
    f"{media_pedidos_dia:.1f} un/dia",
    f"±{desvio_pedidos_dia:.1f} un | {total_pedidos_prod:,} pedidos".replace(",", "."),
)
k2.metric(
    "2. Equação da Demanda",
    f"y = {coef_a_dem:+.2f}x + {coef_b_dem:.1f}",
    f"R² = {r2_dem:.3f} • {tendencia_dem}",
)
k3.metric(
    "3. Tempo Médio (Cenário)",
    f"{tempo_medio_cenario_min:.1f} min",
    f"{diferenca_tempo_min:+.1f} min vs média produto",
    delta_color="inverse",
)
k4.metric(
    "4. Ponto de Reposição (PR)",
    f"{ponto_de_pedido} un.",
    f"ES: +{es_ativo:.1f} un. ({metodo_label_curto})",
)
k5.metric(
    "5. Veículo no Cenário",
    f"{str(melhor_veiculo_cenario).upper()}",
    f"Média: {melhor_tempo_cenario:.1f} min",
)

st.divider()

aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
    "1. Tratamento e Limpeza dos Dados",
    "2a. Regressão Linear e Diagramas de Dispersão",
    "2b. Análise de Sensibilidade (Clima e Trânsito)",
    "2c. Otimização de Frota",
    "3. Ponto de Reposição e Simulação de Estoque",
    "4. Limitações, Premissas e Diagnósticos",
])

# =========================================================
# ABA 1 — TRATAMENTO E LIMPEZA DOS DADOS
# =========================================================
with aba1:
    st.subheader("Etapa 1: Tratamento e Limpeza dos Dados")
    st.info(
        "Base do Kaggle higienizada (`43.739` entregas válidas preservadas, `0` perdas), "
        "com padronização textual, classificação de valores ausentes como `Não Informado`, mediana `4,7` nas notas, "
        "correção de sinal das coordenadas GPS e cálculo da distância pela fórmula de Haversine."
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Apresentação da Base de Dados e Dicionário de Variáveis")
        st.dataframe(pd.DataFrame(DICIONARIO_VARIAVEIS), use_container_width=True, hide_index=True)

    with col_d2:
        st.markdown(f"#### Tabela 3: Parâmetros Consolidados dos 16 Produtos ({'PT-BR' if usar_pt else 'Inglês Original'})")
        st.dataframe(df_parametros, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Tabelas Tratadas (Tabela 1, Tabela 2 e Tabela 3)")
    escolha_tabela = st.radio(
        "Selecione a tabela:",
        options=[
            "Tabela 1: amazon_pedidos_limpo.csv (43.739)",
            "Tabela 2: amazon_estoque_diario.csv (704)",
            "Tabela 3: amazon_parametros_produtos.csv (16)",
        ],
        horizontal=True,
    )
    mapa_tabelas = {
        "Tabela 1: amazon_pedidos_limpo.csv (43.739)": (df_pedidos.drop(columns=["_clima_norm", "_transito_norm"]), "amazon_pedidos_limpo.csv"),
        "Tabela 2: amazon_estoque_diario.csv (704)": (df_estoque, "amazon_estoque_diario.csv"),
        "Tabela 3: amazon_parametros_produtos.csv (16)": (df_parametros, "amazon_parametros_produtos.csv"),
    }
    tabela_vis, nome_csv_exp = mapa_tabelas[escolha_tabela]
    st.write(f"**{len(tabela_vis):,} registros** (prévia limitada a 100 linhas)".replace(",", "."))
    st.dataframe(tabela_vis.head(100), use_container_width=True, hide_index=True)

    st.download_button(
        label=f"Baixar {nome_csv_exp}",
        data=tabela_vis.to_csv(index=False).encode("utf-8-sig"),
        file_name=nome_csv_exp,
        mime="text/csv",
    )

# =========================================================
# ABA 2 — DEMANDA, REGRESSÃO LINEAR E DISPERSÃO
# =========================================================
with aba2:
    st.subheader("Etapa 2a: Regressão Linear e Diagramas de Dispersão")

    with st.expander("O que esta aba analisa e por quê"):
        st.markdown(
            """
- **Analisa:** se a demanda diária de cada produto cresce, cai ou permanece estável ao longo do período, e quanto cada km adicional acrescenta ao tempo de entrega.
- **Modelo:** regressão linear simples $y = a \\cdot x + b$ ajustada por mínimos quadrados (`scipy.stats.linregress`).
- **Como ler:** $R^2$ indica quanto da variação é explicada pela reta (fraco $< 0{,}30$); $p$-valor $< 0{,}05$ indica tendência estatisticamente significativa.
- **Cuidado:** significância estatística não implica causalidade, e sazonalidade semanal não é modelada explicitamente.
            """
        )

    if n_cal != n_dado:
        st.warning(
            f"Há **{n_dado} dias com registros** ao longo de **{n_cal} dias de calendário** "
            f"({n_cal - n_dado} dia(s) sem registro, como 22/03). O eixo $x$ usa posição sequencial contínua (`dia_operacao` 1 a {n_dado})."
        )

    dist_dia = df_estoque.groupby("dia_operacao")[COL_ESTOQUE_DIST].mean().sort_index()
    if len(dist_dia) >= 10:
        autocorr_lag2 = dist_dia.autocorr(lag=2)
        if pd.notna(autocorr_lag2) and autocorr_lag2 > 0.7:
            st.info(
                f"**Nota estrutural da base:** a distância média diária alterna entre ~5 km e ~15 km "
                f"(autocorrelação lag-2 = {autocorr_lag2:.2f}), característica do dataset original. "
                "A regressão de demanda **não é afetada**, pois a quantidade diária de pedidos não alterna."
            )

    st.markdown(f"#### Variação da Demanda Diária ao longo do Tempo — {produto_sel}")
    if len(df_diario_prod) >= 3:
        graf_dem = df_diario_prod[[COL_ESTOQUE_DATA, "dia_operacao", COL_ESTOQUE_DEMANDA]].copy()
        graf_dem["Demanda estimada"] = coef_a_dem * graf_dem["dia_operacao"] + coef_b_dem
        graf_dem["Resíduo"] = graf_dem[COL_ESTOQUE_DEMANDA] - graf_dem["Demanda estimada"]
        resid_std = float(np.std(graf_dem["Resíduo"], ddof=1))
        graf_dem["Sup"] = graf_dem["Demanda estimada"] + 2 * resid_std
        graf_dem["Inf"] = graf_dem["Demanda estimada"] - 2 * resid_std

        st.latex(rf"y = {coef_a_dem:.3f} \cdot x + {coef_b_dem:.2f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Equação", f"y = {coef_a_dem:.3f}x + {coef_b_dem:.2f}")
        m2.metric("R²", f"{r2_dem:.4f}")
        m3.metric("p-valor", formatar_p_valor(p_dem))
        m4.metric("Tendência", tendencia_dem)

        st.caption(
            f"Erro-padrão da inclinação: {se_dem:.4f} | Cada dia adicional altera a demanda média prevista em {coef_a_dem:+.3f} un."
        )
        if r2_dem < 0.30:
            st.info("R² < 0,30: a reta explica pouca variação da demanda diária. Use como descrição de tendência, não como previsão determinística.")

        fig_dem = go.Figure()
        fig_dem.add_trace(go.Scatter(
            x=graf_dem["dia_operacao"], y=graf_dem[COL_ESTOQUE_DEMANDA],
            mode="markers", name="Demanda observada",
            marker=dict(size=8, opacity=0.75),
        ))
        fig_dem.add_trace(go.Scatter(
            x=graf_dem["dia_operacao"], y=graf_dem["Demanda estimada"],
            mode="lines", name="Reta de regressão",
            line=dict(color="firebrick", width=3),
        ))
        fig_dem.add_trace(go.Scatter(
            x=graf_dem["dia_operacao"], y=graf_dem["Sup"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_dem.add_trace(go.Scatter(
            x=graf_dem["dia_operacao"], y=graf_dem["Inf"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(178,34,34,0.12)", name="Faixa ±2σ (resíduos)",
            hoverinfo="skip",
        ))
        fig_dem.update_layout(
            title=f"Diagrama de Dispersão: Demanda Diária × Tempo ({produto_sel})",
            xaxis_title=f"Dia de operação sequencial (1 a {n_dado})",
            yaxis_title="Quantidade de pedidos no dia (y)",
            legend_title_text="",
        )
        st.plotly_chart(fig_dem, use_container_width=True)

        with st.expander("Ver gráfico de resíduos (observado − estimado)"):
            st.caption("Resíduos ideais exibem dispersão aleatória em torno de zero, sem padrão sistemático.")
            fig_res = px.scatter(
                graf_dem, x="dia_operacao", y="Resíduo",
                title=f"Resíduos da Regressão — {produto_sel}",
                labels={"dia_operacao": "Dia de operação", "Resíduo": "Resíduo (un.)"},
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.warning("Poucas observações para ajustar a regressão linear deste produto.")

    st.markdown("#### Resumo da Regressão por Categoria")
    resultados_reg = []
    for prod in sorted(df_estoque[COL_PRODUTO].unique()):
        d_sub = df_estoque[df_estoque[COL_PRODUTO] == prod].sort_values("dia_operacao")
        if len(d_sub) < 3:
            continue
        r = ajustar_regressao(d_sub["dia_operacao"], d_sub[COL_ESTOQUE_DEMANDA])
        resultados_reg.append({
            "Produto": prod,
            "Inclinação (a)": r["a"],
            "Intercepto (b)": r["b"],
            "Equação": f"y = {r['a']:.3f}x + {r['b']:.2f}",
            "R²": r["r2"],
            "p-valor": r["p"],
            "Tendência": classificar_tendencia(r["a"]),
        })
    st.dataframe(pd.DataFrame(resultados_reg).round(4), use_container_width=True, hide_index=True)
    st.caption("p-valor < 0,05 indica tendência linear estatisticamente significativa.")

    st.divider()
    st.markdown("#### Tempo de Entrega em função da Distância")
    with st.expander("O que esta análise mostra e por quê"):
        st.markdown(
            """
- **Analisa:** quantos minutos extras cada quilômetro adicional de distância adiciona ao tempo de entrega.
- **Modelo:** $Tempo = a \\cdot Distância + b$.
- **Observação:** $R^2$ moderado/baixo é esperado porque clima, trânsito e tipo de veículo também impactam fortemente a duração da entrega.
            """
        )

    escopo_dist = st.radio(
        "Escopo da análise Tempo × Distância:",
        options=[
            f"Cenário selecionado ({produto_sel} | Clima: {clima_sel} | Trânsito: {transito_sel})",
            "Base completa (todos os 43.739 pedidos)",
        ],
        horizontal=True,
    )
    df_alvo_dist = df_cenario if escopo_dist.startswith("Cenário") else df_pedidos

    reg_dist = ajustar_regressao(df_alvo_dist[COL_DISTANCIA], df_alvo_dist[COL_TEMPO_MIN])
    a_d, b_d, r2_d, p_d = reg_dist["a"], reg_dist["b"], reg_dist["r2"], reg_dist["p"]

    st.latex(rf"Tempo = {a_d:.2f} \cdot Dist\hat{{a}}ncia + {b_d:.1f}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Equação", f"y = {a_d:.3f}x + {b_d:.2f}")
    d2.metric("R²", f"{r2_d:.4f}")
    d3.metric("p-valor", formatar_p_valor(p_d))

    qual_r2 = "fraco" if r2_d < 0.30 else ("moderado" if r2_d < 0.70 else "forte")
    st.caption(
        f"Cada 1 km adicional acrescenta em média **{a_d:.2f} minutos** ao tempo de entrega, "
        f"partindo de um patamar base fixo de **{b_d:.1f} min** (R² = {r2_d:.3f} — ajuste {qual_r2})."
    )

    amostra_dist = df_alvo_dist.sample(n=min(4000, len(df_alvo_dist)), random_state=42)
    x_linha = np.linspace(float(df_alvo_dist[COL_DISTANCIA].min()), float(df_alvo_dist[COL_DISTANCIA].max()), 150)
    y_linha = a_d * x_linha + b_d

    fig_dist = px.scatter(
        amostra_dist,
        x=COL_DISTANCIA,
        y=COL_TEMPO_MIN,
        opacity=0.35,
        title="Diagrama de Dispersão: Tempo de Entrega × Distância (amostra visual)",
        labels={COL_DISTANCIA: "Distância Loja–Cliente (km)", COL_TEMPO_MIN: "Tempo de entrega (min)"},
    )
    fig_dist.add_scatter(x=x_linha, y=y_linha, mode="lines", name="Reta de regressão", line=dict(color="orange", width=3))
    st.plotly_chart(fig_dist, use_container_width=True)

# =========================================================
# ABA 3 — ANÁLISE DE SENSIBILIDADE (CLIMA E TRÂNSITO)
# =========================================================
with aba3:
    st.subheader("Etapa 2b: Análise de Sensibilidade (Clima e Trânsito)")

    with st.expander("O que esta aba analisa e por quê"):
        st.markdown(
            r"""
- **Analisa:** quanto o tempo médio de entrega piora em cada condição externa comparada ao cenário mais favorável (`Livre` para trânsito e `Ensolarado` para clima).
- **Métricas:** diferença absoluta em minutos ($\Delta$ min) e relativa ($\Delta$ %), que alimentam o **fator de atraso** do ponto de reposição.
- **Cuidado:** comparação descritiva — outras variáveis como distância e frota atuam em conjunto.
            """
        )

    filtrar_sens_prod = st.checkbox(
        f"Analisar sensibilidade apenas para `{produto_sel}` (desmarque para ver toda a base)",
        value=False,
    )
    df_sens = df_prod if filtrar_sens_prod else df_pedidos

    st.markdown("#### Análise de Sensibilidade — Trânsito")
    renderizar_bloco_sensibilidade(
        df_base=df_sens,
        col_grupo=COL_TRANSITO,
        col_tempo=COL_TEMPO_MIN,
        titulo_grupo="Trânsito",
        valores_baseline=["livre", "low"],
        rotulo_baseline="livre",
        escala_cor="Reds",
    )

    st.divider()
    st.markdown("#### Análise de Sensibilidade — Clima")
    renderizar_bloco_sensibilidade(
        df_base=df_sens,
        col_grupo=COL_CLIMA,
        col_tempo=COL_TEMPO_MIN,
        titulo_grupo="Clima",
        valores_baseline=["ensolarado", "sunny"],
        rotulo_baseline="ensolarado",
        escala_cor="Blues",
    )

    st.divider()
    st.markdown("#### Cenário Combinado (Clima × Trânsito)")
    st.caption("Combinações críticas emergem quando clima adverso e trânsito congestionado coincidem.")

    combinado = (
        df_sens.groupby([COL_TRANSITO, COL_CLIMA])[COL_TEMPO_MIN]
        .agg(Tempo_medio="mean", Quantidade="count")
        .reset_index()
    )
    fig_heat = px.density_heatmap(
        combinado,
        x=COL_TRANSITO,
        y=COL_CLIMA,
        z="Tempo_medio",
        histfunc="avg",
        text_auto=".1f",
        title="Mapa de calor do tempo médio de entrega (min)",
        labels={COL_TRANSITO: "Trânsito", COL_CLIMA: "Clima", "Tempo_medio": "Tempo médio (min)"},
        color_continuous_scale="YlOrRd",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(combinado.sort_values("Tempo_medio", ascending=False).round(2), use_container_width=True, hide_index=True)

# =========================================================
# ABA 4 — OTIMIZAÇÃO PRESCRITIVA DE FROTA (VEÍCULOS)
# =========================================================
with aba4:
    st.subheader("Etapa 2c: Otimização de Frota")
    st.markdown("#### Modelo de Decisão: Otimização do Veículo")

    with st.expander("O que esta aba analisa e por quê"):
        st.markdown(
            """
- **Analisa:** desempenho médio de cada tipo de veículo nos diferentes níveis de trânsito.
- **Critério Prescritivo Global:** como a frota atende a todos os cenários operacionais, o critério adequado é o **tempo esperado ponderado** $E[T_v] = \\sum p(c) \\cdot T(v, c)$, ponderado pela frequência real $p(c)$ de cada nível de trânsito.
- **Rigor amostral:** excluir `Mercado (Perecíveis)` (entregas ultrarrápidas ~5× menores) e células com $n < 30$ (como `Bicicleta`, com apenas 15 registros) evita distorções estatísticas.
            """
        )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtrar_mercado = st.checkbox(
            "Excluir Mercado (Perecíveis) — tempo ~5× menor",
            value=True,
            help="Perecíveis têm padrão operacional distinto; misturar distorce a comparação geral da frota.",
        )
    with col_f2:
        apenas_robustas = st.checkbox(
            "Restringir a células veículo × trânsito com n ≥ 30",
            value=True,
            help="Evita recomendar veículos ou cenários com médias estatisticamente frágeis.",
        )
    with col_f3:
        frota_apenas_prod = st.checkbox(
            f"Filtrar apenas produto selecionado ({produto_sel})",
            value=False,
        )

    frota = df_pedidos[~df_pedidos["_transito_norm"].isin(["não informado", "unknown"])].copy()
    if frota_apenas_prod:
        frota = frota[frota[COL_PRODUTO] == produto_sel]
    if filtrar_mercado:
        frota = frota[~frota[COL_PRODUTO].isin(["Mercado (Perecíveis)", "Grocery"])]

    contagem_veic = frota[COL_VEICULO].value_counts()
    veiculos_raros = contagem_veic[contagem_veic < 30].index.tolist()
    if veiculos_raros:
        st.warning(
            "Veículos com amostra insuficiente ($n < 30$, excluídos da otimização): "
            + ", ".join(f"**{v}** ({contagem_veic[v]})" for v in veiculos_raros)
        )
        frota = frota[~frota[COL_VEICULO].isin(veiculos_raros)]

    if apenas_robustas and not frota.empty:
        cont_cel = frota.groupby([COL_VEICULO, COL_TRANSITO]).size()
        celulas_ok = cont_cel[cont_cel >= 30].index
        frota = frota[frota.set_index([COL_VEICULO, COL_TRANSITO]).index.isin(celulas_ok)]

    if frota.empty:
        st.warning("Nenhuma célula atende aos filtros selecionados nesta aba. Desmarque uma das restrições acima para visualizar a matriz completa.")
    else:
        matriz_frota = frota.groupby([COL_VEICULO, COL_TRANSITO])[COL_TEMPO_MIN].mean().unstack()
        st.markdown("**1. Tempo médio (min) — Veículo × Trânsito**")
        st.dataframe(matriz_frota.round(2), use_container_width=True)

        melhores_por_cenario = (
            frota.groupby([COL_TRANSITO, COL_VEICULO])[COL_TEMPO_MIN]
            .agg(Tempo_medio="mean", n="count")
            .reset_index()
            .sort_values("Tempo_medio")
            .groupby(COL_TRANSITO, as_index=False)
            .first()
        )
        st.markdown("**2. Veículo mais rápido em cada cenário de trânsito (visão local)**")
        st.dataframe(melhores_por_cenario.round(2), use_container_width=True, hide_index=True)

        freq_cenario = frota[COL_TRANSITO].value_counts(normalize=True)
        desempenho = calcular_desempenho_ponderado(matriz_frota, freq_cenario)

        st.markdown("**3. Desempenho global ponderado pela frequência real dos cenários $E[T_v]$**")
        st.dataframe(desempenho.round(2), use_container_width=True)

        validos_pond = desempenho["Tempo médio ponderado"].dropna()
        if len(validos_pond) >= 2:
            melhor_geral = str(validos_pond.idxmin())
            pior_geral = str(validos_pond.idxmax())
            ganho = float(validos_pond.loc[pior_geral] - validos_pond.loc[melhor_geral])
            entregas_dia = len(df_pedidos) / max(n_dado, 1)
            horas_ganhas = ganho * entregas_dia / 60.0

            st.success(
                f"**Decisão Prescritiva Global de Frota:** Priorizar **{melhor_geral.upper()}** "
                f"(tempo esperado ponderado de **{validos_pond.loc[melhor_geral]:.1f} min**). "
                f"Ganho médio de **{ganho:.1f} min/entrega** em relação a **{pior_geral}**."
            )
            st.info(
                f"Considerando a média histórica de ~{entregas_dia:.0f} entregas/dia na operação completa, "
                f"a escolha do melhor veículo representa uma economia potencial de **~{horas_ganhas:.1f} horas-veículo/dia**."
            )
        elif len(validos_pond) == 1:
            melhor_geral = str(validos_pond.idxmin())
            st.success(f"**Veículo elegível:** **{melhor_geral.upper()}** ({validos_pond.loc[melhor_geral]:.1f} min).")

        frota_media = frota.groupby([COL_TRANSITO, COL_VEICULO])[COL_TEMPO_MIN].mean().reset_index()
        fig_v = px.bar(
            frota_media,
            x=COL_TRANSITO,
            y=COL_TEMPO_MIN,
            color=COL_VEICULO,
            barmode="group",
            text_auto=".1f",
            title="Comparação de veículos por nível de trânsito",
            labels={COL_TRANSITO: "Trânsito", COL_TEMPO_MIN: "Tempo médio (min)", COL_VEICULO: "Veículo"},
        )
        st.plotly_chart(fig_v, use_container_width=True)

    st.divider()
    st.markdown(f"#### Cenário Específico Filtrado (`{produto_sel}` | Clima: `{clima_sel}` | Trânsito: `{transito_sel}`)")
    fig_veic_cen = px.bar(
        ranking_veiculos_cenario,
        x=COL_VEICULO,
        y="tempo_medio_min",
        text_auto=".1f",
        color=COL_VEICULO,
        title=f"Tempo médio por veículo no cenário selecionado ({produto_sel})",
        labels={COL_VEICULO: "Veículo", "tempo_medio_min": "Tempo médio (min)"},
    )
    st.plotly_chart(fig_veic_cen, use_container_width=True)
    st.caption("Nota: menor média isolada não comprova causalidade; distância da rota, região e horário também influenciam.")

# =========================================================
# ABA 5 — PONTO DE REPOSIÇÃO E SIMULAÇÃO DE ESTOQUE (44 DIAS)
# =========================================================
with aba5:
    st.subheader("Etapa 3: Ponto de Reposição e Simulação de Estoque")
    st.markdown("#### Cálculo do Ponto de Reposição de Estoque")

    st.warning(
        "Os arquivos originais não contêm saldo físico inicial nem prazo real de fornecedor. "
        "Esta aba executa uma **simulação prescritiva** parametrizada pelos controles da barra lateral."
    )

    with st.expander("O que esta aba analisa e por quê"):
        st.markdown(
            r"""
- **Analisa:** em que nível de estoque disparar o pedido de compra para evitar ruptura diante de oscilações da demanda e atrasos logísticos.
- **Modelo Estatístico (Nível de Serviço $z$):**
  - $f_{atraso} = \max\left(\frac{\bar{T}_{cenário}}{\bar{T}_{baseline}} - 1,\, 0\right)$ e $L_{ef} = L \cdot (1 + f_{atraso})$
  - $ES = z \cdot \sigma_d \cdot \sqrt{L_{ef}}$ e $PR = \lceil d \cdot L_{ef} + ES \rceil$
- **Modelo Percentual (Estoque Básico + Reserva):**
  - $PR_{pct} = \lceil (d \cdot L) + (d \cdot L) \cdot \text{margem\%} \cdot (\bar{T}_{cenario\_prod} / \bar{T}_{geral\_prod}) \rceil$
- **Decisão Gerencial:** equilibrar nível de serviço (menor risco de falta) versus capital imobilizado em estoque.
            """
        )

    st.info(
        f"**Cenário selecionado (Trânsito: `{transito_sel}` | Clima: `{clima_sel}`):** "
        f"Tempo médio global do cenário = **{cenario_tempo_global_min:.1f} min** vs. Baseline (`Livre + Ensolarado`) = **{baseline_tempo_min:.1f} min** "
        f"→ Fator de atraso $f_{{atraso}}$ = **{fator_atraso_pct * 100:.1f}%** → Prazo efetivo $L_{{ef}}$ = **{L_efetivo:.2f} dias** (para $L = {prazo_L}$ dias)."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"##### Nível de Serviço (Estatístico) ({'ATIVO' if usar_estatistico else 'Comparativo'}) — `{nivel_servico_sel}`")
        s1, s2, s3 = st.columns(3)
        s1.metric("Demanda no prazo (d·L_ef)", f"{demanda_durante_prazo_ef:.1f} un")
        s2.metric("Estoque de Segurança (ES)", f"{es_estatistico:.1f} un")
        s3.metric("Ponto de Reposição (PR)", f"{pr_estatistico} un")

    with col_m2:
        st.markdown(f"##### Reserva de Segurança (%) ({'ATIVO' if not usar_estatistico else 'Comparativo'}) — `{margem_seguranca_pct}%`")
        p1, p2, p3 = st.columns(3)
        p1.metric("Estoque Básico (d·L)", f"{estoque_basico_pct:.1f} un")
        p2.metric("Reserva ajustada", f"{es_percentual:.1f} un")
        p3.metric("Ponto de Reposição (PR)", f"{pr_percentual} un")

    with st.expander("Ver equações matemáticas aplicadas"):
        st.latex(r"L_{ef} = L \cdot (1 + f_{\text{atraso}})")
        st.latex(r"ES_{\text{estatístico}} = z \cdot \sigma_d \cdot \sqrt{L_{ef}} \qquad PR_{\text{estatístico}} = \lceil d \cdot L_{ef} + ES_{\text{estatístico}} \rceil")
        st.latex(r"PR_{\text{percentual}} = \lceil (d \cdot L) + (d \cdot L) \cdot p_{\text{reserva}} \cdot f_{\text{rel\_prod}} \rceil")

    st.divider()
    st.markdown(f"#### Simulação do Comportamento do Estoque ao longo dos {n_dado} Dias — {produto_sel}")
    st.caption(
        f"Gatilho ativo: **PR = {ponto_de_pedido} un.** | Estoque inicial: **{max(ponto_de_pedido * 2, lote_reposicao)} un.** | "
        f"Lote de reposição: **+{lote_reposicao} un.** (1,5x a demanda média no prazo)."
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_est_disp = px.scatter(
            df_sim_real,
            x="dia_numero",
            y="estoque_fim_do_dia",
            color="acionou_reposicao",
            size="demanda_real_dia",
            title=f"Dispersão do Estoque ao Longo do Tempo ({produto_sel})",
            labels={
                "dia_numero": f"Dia de operação (1 a {n_dado})",
                "estoque_fim_do_dia": "Estoque fim do dia (un.)",
                "acionou_reposicao": "Reposição?",
            },
        )
        fig_est_disp.add_hline(
            y=ponto_de_pedido,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"PR ({ponto_de_pedido} un)",
        )
        st.plotly_chart(fig_est_disp, use_container_width=True)

    with col_e2:
        fig_est_linha = px.line(
            df_sim_real,
            x="data",
            y=["estoque_fim_do_dia", "demanda_real_dia", "demanda_regressao_linear"],
            markers=True,
            title=f"Trajetória Contínua: Estoque vs. Demanda ({produto_sel})",
            labels={"value": "Unidades", "variable": "Indicador", "data": "Data"},
        )
        fig_est_linha.add_hline(
            y=ponto_de_pedido,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"PR ({ponto_de_pedido} un)",
        )
        st.plotly_chart(fig_est_linha, use_container_width=True)

    with st.expander("Ver tabela diária da simulação (44 dias)"):
        st.dataframe(df_sim_real, use_container_width=True, hide_index=True)

    st.markdown("#### Recomendação Prescritiva Consolidada")
    sim_r1, sim_r2, sim_r3, sim_r4 = st.columns(4)
    sim_r1.metric("Reposições acionadas", f"{total_reposicoes_sim} vezes")
    sim_r2.metric("Dias com ruptura (saldo 0)", f"{dias_ruptura_sim} dia(s)")
    sim_r3.metric("Estoque médio simulado", f"{estoque_medio_sim:.1f} un.")
    sim_r4.metric("Menor saldo registrado", f"{estoque_min_sim} un.")

    status_ruptura_txt = (
        "sem ocorrência de ruptura de estoque"
        if dias_ruptura_sim == 0
        else f"com {dias_ruptura_sim} dia(s) em ruptura — recomenda-se aumentar o nível de serviço, a margem de segurança ou o lote de compra"
    )
    st.success(
        f"**Prescrição para `{produto_sel}` (Clima: `{clima_sel}` | Trânsito: `{transito_sel}`):**\n"
        f"- **Quando comprar:** acionar reposição sempre que o estoque atingir **{ponto_de_pedido} unidades** "
        f"(modelo ativo: {metodo_es}, com reserva de segurança de **{es_ativo:.1f} unidades**).\n"
        f"- **Quanto comprar por pedido:** lote sugerido de **+{lote_reposicao} unidades** (1,5x a demanda média no prazo).\n"
        f"- **Resultado nos {n_dado} dias simulados:** **{total_reposicoes_sim} pedido(s) de reposição** acionados, "
        f"saldo médio de **{estoque_medio_sim:.1f} un.** e {status_ruptura_txt}.\n"
        f"- **Qual veículo alocar:** priorizar **{str(melhor_veiculo_cenario).upper()}** neste cenário "
        f"(tempo médio de **{melhor_tempo_cenario:.1f} min**)."
    )

    st.divider()
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("#### Tabela de Trade-off (Nível de Serviço × Tempo de Reposição)")
        st.caption("Mais nível de serviço ou maior prazo L exige mais capital imobilizado.")
        linhas_tradeoff = []
        for nome_nivel, z_val in NIVEIS_SERVICO_Z.items():
            for L_teste in [1, 3, 7, 14]:
                L_test_ef = L_teste * (1.0 + fator_atraso_pct)
                pr_medio = (
                    df_parametros[COL_PARAM_MEDIA] * L_test_ef
                    + z_val * df_parametros[COL_PARAM_DESVIO] * math.sqrt(L_test_ef)
                ).mean()
                linhas_tradeoff.append({
                    "Nível de serviço": nome_nivel.split(" ")[0],
                    "Prazo L (dias)": L_teste,
                    "L_ef (dias)": round(L_test_ef, 2),
                    "PR médio (16 prod.)": round(float(pr_medio), 1),
                })
        st.dataframe(pd.DataFrame(linhas_tradeoff), use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("#### Ponto de Reposição para as 16 Categorias de Produtos")
        sim_todos = df_parametros[[COL_PRODUTO, COL_PARAM_MEDIA, COL_PARAM_DESVIO]].copy()
        sim_todos = sim_todos.rename(columns={
            COL_PRODUTO: "Produto",
            COL_PARAM_MEDIA: "Demanda média/dia",
            COL_PARAM_DESVIO: "Desvio padrão",
        })
        sim_todos["Demanda no prazo (L_ef)"] = sim_todos["Demanda média/dia"] * L_efetivo
        sim_todos["ES Estatístico"] = z_score * sim_todos["Desvio padrão"] * math.sqrt(L_efetivo)
        sim_todos["PR Estatístico (un.)"] = np.ceil(
            sim_todos["Demanda no prazo (L_ef)"] + sim_todos["ES Estatístico"]
        ).astype(int)
        sim_todos["PR Percentual (un.)"] = np.ceil(
            sim_todos["Demanda média/dia"] * prazo_L * (1.0 + (margem_seguranca_pct / 100.0) * (1.0 + fator_atraso_pct))
        ).astype(int)

        st.dataframe(sim_todos.round(2), use_container_width=True, hide_index=True)
        st.download_button(
            label="Baixar simulação dos 16 produtos (CSV)",
            data=sim_todos.round(2).to_csv(index=False).encode("utf-8-sig"),
            file_name="ponto_reposicao_simulado.csv",
            mime="text/csv",
        )

# =========================================================
# ABA 6 — DIAGNÓSTICOS ESTATÍSTICOS E LIMITAÇÕES
# =========================================================
with aba6:
    st.subheader("Limitações, Premissas do Modelo e Diagnósticos Estatísticos")
    st.markdown(
        "Verificação formal das hipóteses estatísticas utilizadas nas etapas anteriores. "
        "Identificar desvios de normalidade, autocorrelação ou baixa cobertura amostral garante rigor técnico na interpretação."
    )

    st.markdown("**1) Volatilidade da Demanda por Produto (CV e Sobredispersão)**")
    diag_vol = df_parametros.copy()
    diag_vol["CV (%)"] = (diag_vol[COL_PARAM_DESVIO] / diag_vol[COL_PARAM_MEDIA] * 100.0).round(1)
    diag_vol["σ / √μ"] = (diag_vol[COL_PARAM_DESVIO] / np.sqrt(diag_vol[COL_PARAM_MEDIA])).round(2)
    st.dataframe(
        diag_vol[[COL_PRODUTO, COL_PARAM_MEDIA, COL_PARAM_DESVIO, "CV (%)", "σ / √μ"]]
        .sort_values("CV (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("CV elevado indica maior variabilidade relativa e exige reserva de segurança proporcionalmente maior; σ/√μ > 1,3 indica sobredispersão.")

    st.markdown("**2) Teste de Normalidade (Shapiro-Wilk) e Autocorrelação Lag-1**")
    diag_norm = []
    for prod in sorted(df_estoque[COL_PRODUTO].unique()):
        serie = (
            df_estoque[df_estoque[COL_PRODUTO] == prod]
            .sort_values("dia_operacao")[COL_ESTOQUE_DEMANDA]
            .dropna()
            .values
        )
        if len(serie) < 8:
            continue
        if HAS_SCIPY:
            try:
                _, p_norm = stats.shapiro(serie)
            except Exception:
                p_norm = None
        else:
            p_norm = None

        r_lag1 = None
        if len(serie) >= 3 and np.std(serie[:-1]) > 0 and np.std(serie[1:]) > 0:
            r_lag1 = float(np.corrcoef(serie[:-1], serie[1:])[0, 1])

        diag_norm.append({
            "Produto": prod,
            "n": len(serie),
            "p Shapiro-Wilk": round(float(p_norm), 4) if p_norm is not None else None,
            "Normal? (p ≥ 0,05)": "Sim" if p_norm is not None and p_norm >= 0.05 else ("Não" if p_norm is not None else "—"),
            "r lag-1": round(r_lag1, 3) if r_lag1 is not None else None,
            "Independente? (|r| < 0,3)": "Sim" if r_lag1 is not None and abs(r_lag1) < 0.3 else ("Não" if r_lag1 is not None else "—"),
        })

    if diag_norm:
        st.dataframe(pd.DataFrame(diag_norm), use_container_width=True, hide_index=True)
        n_rej = sum(1 for item in diag_norm if item["Normal? (p ≥ 0,05)"] == "Não")
        n_auto = sum(1 for item in diag_norm if item["Independente? (|r| < 0,3)"] == "Não")
        if n_rej > 0:
            st.warning(f"{n_rej}/{len(diag_norm)} categorias rejeitam normalidade (p < 0,05) — para elas, o quantil empírico é um complemento recomendado ao z-score.")
        if n_auto > 0:
            st.warning(f"{n_auto} categoria(s) apresentam autocorrelação relevante (|r| ≥ 0,3) — a aproximação por √L tende a subestimar o estoque de segurança.")
        if n_rej == 0 and n_auto == 0:
            st.success("As hipóteses de normalidade e independência diária sustentam-se em todas as categorias.")
    else:
        st.info("Dados insuficientes para os testes de normalidade.")

    col_d_a, col_d_b = st.columns(2)
    with col_d_a:
        st.markdown("**3) Cobertura Amostral: Veículo × Trânsito**")
        cont_cel_diag = df_pedidos.groupby([COL_VEICULO, COL_TRANSITO]).size().unstack(fill_value=0)
        st.dataframe(cont_cel_diag, use_container_width=True)
        frag = int((cont_cel_diag < 30).sum().sum())
        st.caption(f"{frag} de {cont_cel_diag.size} células têm n < 30; médias nessas células são estatisticamente frágeis.")

    with col_d_b:
        st.markdown("**4) Proporção de Registros 'Não Informado' / 'Unknown'**")
        ni_rows = []
        for col in [COL_CLIMA, COL_TRANSITO, COL_VEICULO, COL_REGIAO]:
            total = len(df_pedidos)
            n_ni = int(df_pedidos[col].astype(str).str.strip().str.casefold().isin(["não informado", "unknown"]).sum())
            ni_rows.append({"Variável": col, "Registros": n_ni, "% do total": round(100.0 * n_ni / total, 2)})
        st.dataframe(pd.DataFrame(ni_rows), use_container_width=True, hide_index=True)

        st.markdown("**5) Distribuição da Frota**")
        fr = df_pedidos[COL_VEICULO].value_counts().reset_index()
        fr.columns = ["Veículo", "Pedidos"]
        fr["%"] = (fr["Pedidos"] / len(df_pedidos) * 100.0).round(2)
        fr["Status"] = fr["Pedidos"].apply(lambda n: "OK" if n >= 30 else "Insuficiente (n < 30)")
        st.dataframe(fr, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Projeto acadêmico de Análise de Dados Prescritiva (UNISO • 2026). "
    "Os resultados refletem os dados históricos de 44 dias de operação e dependem das premissas configuradas; "
    "regressão linear e comparações descritivas não comprovam causalidade isolada."
)
