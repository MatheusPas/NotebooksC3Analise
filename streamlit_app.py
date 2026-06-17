import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import Connection as conn

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.facecolor': '#111111',
    'axes.facecolor': '#111111',
    'axes.edgecolor': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'text.color': '#f2f2f2',
    'grid.color': '#444444',
    'figure.edgecolor': '#111111',
    'savefig.facecolor': '#111111',
    'savefig.edgecolor': '#111111',
})


def set_streamlit_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #0f0f0f;
            color: #f2f2f2;
        }
        .stApp {
            background-color: #0f0f0f;
            color: #f2f2f2;
        }
        .css-1d391kg {
            background-color: #111111;
        }
        .css-1d391kg .stButton>button {
            background-color: #c1121f;
            color: #ffffff;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #ffffff;
        }
        .stMarkdown a {
            color: #f2f2f2;
        }
        .stSidebar {
            background-color: #111111;
        }
        .css-12oz5g7 {
            background-color: #111111;
        }
        .stButton > button {
            background-color: #c1121f;
            color: #ffffff;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, anchor: str):
    st.markdown(f'<a id="{anchor}"></a>\n<h2 style="color:#ffffff; font-family: Arial, sans-serif;">{title}</h2>', unsafe_allow_html=True)


def sidebar_index():
    menu = [
        ('Visão Geral', '#visao-geral'),
        ('Feature Engineering', '#feature-engineering'),
        ('Regressão Linear', '#regressao'),
        ('Classificação Binária', '#classificacao'),
        ('PCA e Clusterização', '#pca-clusterizacao'),
        ('API / Endpoints', '#api-endpoints'),
    ]
    st.sidebar.markdown('### Índice', unsafe_allow_html=True)
    for name, href in menu:
        st.sidebar.markdown(f'- [{name}]({href})', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_api_headers(username: str, password: str):
    try:
        headers = conn.get_conn(username, password)
        if not headers or 'Authorization' not in headers or 'Bearer None' in headers['Authorization']:
            return None
        return headers
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dataset(headers):
    if not headers:
        return None, None
    try:
        df, df_tratado = conn.get_dataset_completo(headers)
        return df, df_tratado
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_api_info(headers):
    if not headers:
        return {}
    info = {}
    try:
        info['stats'] = conn.get_stats(headers)
    except Exception:
        info['stats'] = None
    try:
        info['missing'] = conn.get_missing(headers)
    except Exception:
        info['missing'] = None
    try:
        info['distribution'] = conn.get_distribution(headers)
    except Exception:
        info['distribution'] = None
    try:
        info['correlations'] = conn.get_correlations(headers)
    except Exception:
        info['correlations'] = None
    try:
        info['reg_simple'] = conn.get_regression_simple(headers)
    except Exception:
        info['reg_simple'] = None
    try:
        info['reg_multiple'] = conn.get_regression_multiple(headers)
    except Exception:
        info['reg_multiple'] = None
    try:
        info['kmeans'] = conn.get_kmeans(headers, n_clusters=3)
    except Exception:
        info['kmeans'] = None
    return info


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'house_age' not in df.columns and 'year_built' in df.columns:
        df['house_age'] = 2024 - df['year_built']
    if 'remod_age' not in df.columns and 'year_remod_add' in df.columns:
        df['remod_age'] = 2024 - df['year_remod_add']
    if 'total_bath' not in df.columns and {'full_bath', 'half_bath'}.issubset(df.columns):
        df['total_bath'] = df['full_bath'] + 0.5 * df['half_bath']
    if 'area_per_room' not in df.columns and {'gr_liv_area', 'totrms_abvgrd'}.issubset(df.columns):
        df['area_per_room'] = df['gr_liv_area'] / (df['totrms_abvgrd'].replace(0, np.nan) + 1)
    if 'price_category' not in df.columns and 'sale_price' in df.columns:
        median_price = df['sale_price'].median()
        df['price_category'] = (df['sale_price'] > median_price).astype(int)
    return df


def show_dataset_overview(df: pd.DataFrame, info: dict):
    section_title('Visão Geral do Dataset', 'visao-geral')
    st.write('Dados coletados da API e processados localmente para análise e visualização.')
    st.subheader('Amostra dos dados')
    st.dataframe(df.head(10), use_container_width=True)

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    st.markdown(f'- Colunas numéricas: **{len(num_cols)}**')
    st.markdown(f'- Colunas categóricas: **{len(cat_cols)}**')
    st.markdown(f'- Total de linhas: **{df.shape[0]}**')
    st.markdown(f'- Total de colunas: **{df.shape[1]}**')

    if info.get('stats'):
        st.subheader('Resumo estatístico da API')
        st.json(info['stats'])

    st.subheader('Valores faltantes')
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success('Não há valores faltantes no dataset carregado.')
    else:
        st.dataframe(missing.to_frame('missing_count'))
        fig, ax = plt.subplots(figsize=(7, 3))
        missing.plot(kind='bar', ax=ax, color='#c1121f', edgecolor='white')
        ax.set_title('Valores Faltantes por Coluna')
        ax.set_ylabel('Quantidade de Missing')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    if 'sale_price' in df.columns:
        st.subheader('Distribuição do preço de venda')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].hist(df['sale_price'].dropna(), bins=40, color='#c1121f', edgecolor='white')
        axes[0].set_title('Histograma do Preço de Venda')
        axes[0].set_xlabel('sale_price')
        axes[0].set_ylabel('Frequência')
        if info.get('distribution'):
            dist = info['distribution'].get('distribuicao_preco', {})
            media = dist.get('media')
            mediana = dist.get('mediana')
            if media is not None:
                axes[0].axvline(media, color='red', linestyle='--', label='Média')
            if mediana is not None:
                axes[0].axvline(mediana, color='green', linestyle='--', label='Mediana')
            axes[0].legend()
        axes[1].boxplot(df['sale_price'].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[1].set_title('Boxplot do Preço de Venda')
        axes[1].set_xticks([])
        st.pyplot(fig)

    with st.expander('Correlação entre variáveis numéricas'):
        numeric_df = df.select_dtypes(include='number').drop(columns=['id'], errors='ignore')
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdGy', center=0,
                        linewidths=0.4, cbar_kws={'shrink': 0.7}, ax=ax)
            ax.set_title('Heatmap de Correlação')
            st.pyplot(fig)
        else:
            st.write('Não há variáveis numéricas suficientes para calcular correlação.')

    if info.get('correlations') and isinstance(info['correlations'], dict):
        st.subheader('Correlação com o preço de venda (API)')
        st.json(info['correlations'].get('correlacao_com_preco', {}))


def show_feature_engineering(df: pd.DataFrame):
    section_title('Feature Engineering', 'feature-engineering')
    st.write('Criamos novas variáveis a partir do dataset para enriquecer a análise.')
    st.write('Novas features geradas: house_age, remod_age, total_bath, area_per_room, price_category.')
    st.dataframe(df[['house_age', 'remod_age', 'total_bath', 'area_per_room', 'price_category']].head(10), use_container_width=True)

    st.subheader('Resumo das novas variáveis')
    st.dataframe(df[['house_age', 'remod_age', 'total_bath', 'area_per_room']].describe().round(2))

    if 'price_category' in df.columns:
        st.subheader('Distribuição de price_category')
        counts = df['price_category'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        counts.plot(kind='bar', color=['#ffffff', '#c1121f'], ax=ax)
        ax.set_xlabel('price_category')
        ax.set_ylabel('Contagem')
        ax.set_title('Distribuição da classe de preço')
        ax.set_xticklabels(['Baixo (0)', 'Alto (1)'], rotation=0)
        st.pyplot(fig)


def show_regression(df: pd.DataFrame, info: dict):
    section_title('Regressão Linear', 'regressao')
    st.write('Modelos de regressão linear simples e múltipla usando o dataset de casas.')

    features_simple = ['gr_liv_area']
    features_mult = [
        'lot_area', 'overall_qual', 'overall_cond', 'year_built',
        'gr_liv_area', 'full_bath', 'half_bath', 'bedroom_abvgr',
        'totrms_abvgrd', 'garage_cars', 'garage_area'
    ]
    target = 'sale_price'
    df_reg = df.copy().dropna(subset=features_mult + [target])

    if df_reg.empty:
        st.warning('Não há dados suficientes para treinar os modelos de regressão.')
        return

    X_simple = df_reg[features_simple]
    X_mult = df_reg[features_mult]
    y = df_reg[target]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mult, y, test_size=0.2, random_state=42)

    model_simple = LinearRegression().fit(X_train_s, y_train_s)
    model_mult = LinearRegression().fit(X_train_m, y_train_m)

    y_pred_s = model_simple.predict(X_test_s)
    y_pred_m = model_mult.predict(X_test_m)

    rmse_s = np.sqrt(mean_squared_error(y_test_s, y_pred_s))
    mae_s = mean_absolute_error(y_test_s, y_pred_s)
    r2_s = r2_score(y_test_s, y_pred_s)

    rmse_m = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
    mae_m = mean_absolute_error(y_test_m, y_pred_m)
    r2_m = r2_score(y_test_m, y_pred_m)

    st.subheader('Métricas de desempenho')
    metric_df = pd.DataFrame({
        'Métrica': ['R²', 'MAE', 'RMSE'],
        'Simples (gr_liv_area)': [f'{r2_s:.4f}', f'${mae_s:,.2f}', f'${rmse_s:,.2f}'],
        'Múltipla': [f'{r2_m:.4f}', f'${mae_m:,.2f}', f'${rmse_m:,.2f}'],
    })
    st.table(metric_df)

    st.subheader('Regressão Linear Simples')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(X_test_s, y_test_s, color='#ffffff', alpha=0.6, label='Real')
    ax.plot(X_test_s, y_pred_s, color='#c1121f', linewidth=2, label='Predito')
    ax.set_xlabel('gr_liv_area')
    ax.set_ylabel('sale_price')
    ax.set_title('Regressão Linear Simples')
    ax.legend()
    st.pyplot(fig)

    st.subheader('Regressão Linear Múltipla')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(y_test_m, y_pred_m, color='#ffffff', edgecolors='black', alpha=0.6, s=30)
    min_val = min(y_test_m.min(), y_pred_m.min())
    max_val = max(y_test_m.max(), y_pred_m.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    ax.set_xlabel('Real')
    ax.set_ylabel('Predito')
    ax.set_title('Real vs. Predito — Regressão Múltipla')
    st.pyplot(fig)

    if info.get('reg_simple') or info.get('reg_multiple'):
        with st.expander('Comparação com endpoints de regressão da API'):
            if info.get('reg_simple'):
                st.subheader('Regressão Simples — API')
                st.json(info['reg_simple'])
            if info.get('reg_multiple'):
                st.subheader('Regressão Múltipla — API')
                st.json(info['reg_multiple'])


def show_classification(df: pd.DataFrame):
    section_title('Classificação Binária', 'classificacao')
    if 'price_category' not in df.columns:
        st.warning('A variável price_category não está disponível no dataset.')
        return

    features = ['gr_liv_area', 'overall_qual', 'house_age', 'total_bath', 'garage_cars', 'area_per_room']
    required = [col for col in features + ['price_category'] if col not in df.columns]
    if required:
        st.warning(f'Dados insuficientes para classificação: faltam colunas {required}')
        return

    df_cls = df[features + ['price_category']].dropna()
    X = df_cls[features]
    y = df_cls['price_category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KMeans
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    knn_model = KNeighborsClassifier(n_neighbors=5)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)

    knn_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test)

    def compute_metrics(name, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }

    metrics_knn = compute_metrics('KNN', y_test, y_pred_knn)
    metrics_rf = compute_metrics('Random Forest', y_test, y_pred_rf)

    st.subheader('Métricas dos modelos')
    metrics_df = pd.DataFrame([metrics_knn, metrics_rf], index=['KNN', 'Random Forest']).T
    metrics_df = metrics_df.style.format({col: '{:.4f}' for col in metrics_df.columns})
    st.dataframe(metrics_df)

    st.subheader('Matriz de Confusão — KNN')
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title('KNN — Matriz de Confusão')
    st.pyplot(fig)

    st.subheader('Matriz de Confusão — Random Forest')
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    ax.set_title('Random Forest — Matriz de Confusão')
    st.pyplot(fig)


def show_pca_and_clustering(df: pd.DataFrame, info: dict):
    section_title('PCA e Clusterização', 'pca-clusterizacao')
    numeric = df.select_dtypes(include='number').drop(columns=['id', 'sale_price'], errors='ignore')
    numeric = numeric.dropna(axis=1, how='any')
    if numeric.shape[1] < 2:
        st.warning('Não há variáveis numéricas suficientes para PCA e clusterização.')
        return

    X = numeric.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    st.subheader('Variância explicada pelo PCA')
    explained_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained))],
        'Variância (%)': (explained * 100).round(2),
        'Variância Acumulada (%)': (cumulative * 100).round(2),
    })
    st.dataframe(explained_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
    ax.axhline(0.90, color='orange', linestyle='--', label='90%')
    ax.axhline(0.95, color='red', linestyle='--', label='95%')
    ax.set_xlabel('Número de componentes')
    ax.set_ylabel('Variância acumulada')
    ax.set_title('Variância explicada acumulada pelo PCA')
    ax.legend()
    st.pyplot(fig)

    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(X_pca2[:, 0], X_pca2[:, 1], c=df.loc[X.index, 'sale_price'] if 'sale_price' in df.columns else None,
                         cmap='RdGy', alpha=0.7, s=22)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('sale_price')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA 2D colorido por preço')
    st.pyplot(fig)

    st.subheader('Análise de clusterização K-Means')
    k = st.selectbox('Número de clusters (K)', [2, 3, 4, 5], index=1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], alpha=0.6, s=26, label=f'Cluster {cluster}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=150, label='Centróides')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'K-Means com K={k} — Espaço PCA')
    ax.legend()
    st.pyplot(fig)

    cluster_df = pd.DataFrame(X, columns=X.columns)
    cluster_df['cluster'] = labels
    cluster_df['sale_price'] = df.loc[X.index, 'sale_price'] if 'sale_price' in df.columns else np.nan
    st.subheader('Resumo dos clusters')
    st.dataframe(cluster_df.groupby('cluster').agg({
        'sale_price': ['mean', 'median', 'count'],
        'gr_liv_area': 'mean' if 'gr_liv_area' in cluster_df.columns else 'first',
        'overall_qual': 'mean' if 'overall_qual' in cluster_df.columns else 'first',
    }).round(2))

    if info.get('kmeans'):
        with st.expander('Dados de clusterização retornados pela API'):
            st.json(info['kmeans'])


def main():
    st.set_page_config(page_title='Análise de Casas - Streamlit', layout='wide')
    set_streamlit_style()
    st.markdown('<h1 style="color: #ffffff; font-family: Arial, sans-serif;">Análise de Preços de Casas - Painel Executivo</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #d9d9d9; font-family: Arial, sans-serif; font-size: 16px; margin-top: -15px;">Visualização consolidada de dados, gráficos e relatórios, com paleta corporativa preto, branco, vermelho e cinza.</p>', unsafe_allow_html=True)

    sidebar_index()
    st.markdown('---')
    st.subheader('Conexão com a API')
    username = st.text_input('Usuário', value='matheus_root')
    password = st.text_input('Senha', value='server@Database', type='password')
    action = st.button('Carregar dados', use_container_width=True)

    if 'loaded' not in st.session_state:
        st.session_state.loaded = False

    if action:
        headers = get_api_headers(username, password)
        if headers is None:
            st.error('Falha de autenticação. Verifique usuário e senha.')
            st.session_state.loaded = False
        else:
            df, df_tratado = load_dataset(headers)
            if df is None:
                st.error('Não foi possível carregar o dataset da API.')
                st.session_state.loaded = False
            else:
                st.session_state.loaded = True
                st.session_state.headers = headers
                st.session_state.df = df
                st.session_state.df_tratado = df_tratado
                st.session_state.info = load_api_info(headers)

    if not st.session_state.loaded:
        st.warning('Clique em **Carregar dados** para iniciar a visualização na mesma tela.')
        return

    df = build_features(st.session_state.df)
    info = st.session_state.info or {}

    show_dataset_overview(df, info)
    st.markdown('---')
    show_feature_engineering(df)
    st.markdown('---')
    show_regression(df, info)
    st.markdown('---')
    show_classification(df)
    st.markdown('---')
    show_pca_and_clustering(df, info)
    st.markdown('---')
    section_title('API / Endpoints', 'api-endpoints')
    st.write('Dados retornados pelos endpoints disponíveis na API.')
    st.json(info)


if __name__ == '__main__':
    main()
