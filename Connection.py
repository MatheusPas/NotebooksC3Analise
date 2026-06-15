import requests
import pandas as pd

BASE_URL = "http://26.246.114.225:8000"


def get_conn(username, password):
    """Faz login na API e retorna os headers de autorização.

    Endpoint: POST /auth/login
    Args:
        username: nome de usuário da conta da API.
        password: senha da conta da API.
    Returns:
        dict: {'Authorization': 'Bearer <token>'}
    """
    response = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    token = response.json().get("access_token")
    return {"Authorization": f"Bearer {token}"}

def puxar_df(headers):
    """Busca o dataset completo de casas e transforma em DataFrame.

    Endpoint: GET /api/v1/houses?limit=1460
    Args:
        headers: cabeçalhos de autenticação retornados por get_conn.
    Returns:
        pandas.DataFrame com todas as linhas de casas carregadas.
    """
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/houses?limit=1460", headers=headers).json()
    )

def get_stats(headers):
    """Retorna resumo estatístico do dataset de casas.

    Endpoint: GET /api/v1/houses/stats/summary
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com estatísticas agregadas do conjunto de dados.
    """
    return requests.get(f"{BASE_URL}/api/v1/houses/stats/summary", headers=headers).json()

def get_missing(headers):
    """Retorna a análise de valores faltantes do dataset.

    Endpoint: GET /api/v1/eda/missing-values
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com contagem de missing values por coluna.
    """
    return requests.get(f"{BASE_URL}/api/v1/eda/missing-values", headers=headers).json()

def get_correlations(headers):
    """Retorna a matriz de correlação entre as variáveis.

    Endpoint: GET /api/v1/eda/correlations
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com correlações entre os campos do dataset.
    """
    return requests.get(f"{BASE_URL}/api/v1/eda/correlations", headers=headers).json()

def get_distribution(headers):
    """Retorna análise de distribuição das variáveis.

    Endpoint: GET /api/v1/eda/distribution
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com informações de distribuição para variáveis.
    """
    return requests.get(f"{BASE_URL}/api/v1/eda/distribution", headers=headers).json()

def get_new_features(headers):
    """Retorna um DataFrame contendo novas features geradas pela API.

    Endpoint: GET /api/v1/features/new-features
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        pandas.DataFrame com o campo 'novas_features'.
    """
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/features/new-features", headers=headers).json()["novas_features"]
    )

def get_normalized(headers, method="minmax"):
    """Retorna um DataFrame com dados normalizados.

    Endpoint: GET /api/v1/features/normalize?method={method}
    Args:
        headers: cabeçalhos de autenticação.
        method: método de normalização ('minmax' por padrão).
    Returns:
        pandas.DataFrame com o campo 'amostra_normalizada'.
    """
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/features/normalize?method={method}", headers=headers).json()["amostra_normalizada"]
    )

def get_encoded(headers):
    """Retorna as features codificadas pela API.

    Endpoint: GET /api/v1/features/encode
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com as variáveis categóricas codificadas.
    """
    return requests.get(f"{BASE_URL}/api/v1/features/encode", headers=headers).json()

def get_regression_simple(headers):
    """Retorna resultados da regressão linear simples.

    Endpoint: GET /api/v1/regression/simple
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com métricas e saída do modelo de regressão simples.
    """
    return requests.get(f"{BASE_URL}/api/v1/regression/simple", headers=headers).json()

def get_regression_multiple(headers):
    """Retorna resultados da regressão linear múltipla.

    Endpoint: GET /api/v1/regression/multiple
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com métricas e saída do modelo de regressão múltipla.
    """
    return requests.get(f"{BASE_URL}/api/v1/regression/multiple", headers=headers).json()

def get_classification_knn(headers, k=5):
    """Retorna resultados de classificação KNN.

    Endpoint: GET /api/v1/classification/knn?k={k}
    Args:
        headers: cabeçalhos de autenticação.
        k: número de vizinhos a ser usado.
    Returns:
        dict/JSON com métricas do modelo KNN.
    """
    return requests.get(f"{BASE_URL}/api/v1/classification/knn?k={k}", headers=headers).json()

def get_classification_rf(headers):
    """Retorna resultados de classificação por Random Forest.

    Endpoint: GET /api/v1/classification/random-forest
    Args:
        headers: cabeçalhos de autenticação.
    Returns:
        dict/JSON com métricas do modelo Random Forest.
    """
    return requests.get(f"{BASE_URL}/api/v1/classification/random-forest", headers=headers).json()

def get_kmeans(headers, n_clusters=3):
    """Retorna resultados de clusterização K-Means.

    Endpoint: GET /api/v1/clustering/kmeans?n_clusters={n_clusters}
    Args:
        headers: cabeçalhos de autenticação.
        n_clusters: número de clusters desejado.
    Returns:
        dict/JSON com rótulos de cluster e informações do K-Means.
    """
    return requests.get(f"{BASE_URL}/api/v1/clustering/kmeans?n_clusters={n_clusters}", headers=headers).json()

def get_outliers(headers, n_neighbors=20):
    """Retorna detecção de outliers pelo endpoint de clusterização.

    Endpoint: GET /api/v1/clustering/outliers?n_neighbors={n_neighbors}
    Args:
        headers: cabeçalhos de autenticação.
        n_neighbors: número de vizinhos usado na detecção de outliers.
    Returns:
        dict/JSON com os outliers identificados.
    """
    return requests.get(f"{BASE_URL}/api/v1/clustering/outliers?n_neighbors={n_neighbors}", headers=headers).json()

def get_pca(headers, n_components=2):
    """Retorna resultados da análise de PCA.

    Endpoint: GET /api/v1/pca/run?n_components={n_components}
    Args:
        headers: cabeçalhos de autenticação.
        n_components: número de componentes principais.
    Returns:
        dict/JSON com componentes e variância explicada.
    """
    return requests.get(f"{BASE_URL}/api/v1/pca/run?n_components={n_components}", headers=headers).json()

def get_association(headers, min_support=0.1):
    """Retorna regras de associação geradas pela API.

    Endpoint: GET /api/v1/pca/association?min_support={min_support}
    Args:
        headers: cabeçalhos de autenticação.
        min_support: suporte mínimo para as regras de associação.
    Returns:
        dict/JSON com regras de associação e métricas como suporte e confiança.
    """
    return requests.get(f"{BASE_URL}/api/v1/pca/association?min_support={min_support}", headers=headers).json()

def get_dataset_completo(headers):
    data = requests.get(f"{BASE_URL}/api/v1/features/dataset-completo", headers=headers).json()
    return (
        pd.DataFrame(data["dataset"]),
        pd.DataFrame(data["dataset_tratado"])
    )