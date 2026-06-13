import requests
import pandas as pd

BASE_URL = "http://26.246.114.225:8000"

def get_conn(username, password):
    response = requests.post(f"{BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    token = response.json().get("access_token")
    return {"Authorization": f"Bearer {token}"}

def puxar_df(headers):
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/houses?limit=1460", headers=headers).json()
    )

def get_stats(headers):
    return requests.get(f"{BASE_URL}/api/v1/houses/stats/summary", headers=headers).json()

def get_missing(headers):
    return requests.get(f"{BASE_URL}/api/v1/eda/missing-values", headers=headers).json()

def get_correlations(headers):
    return requests.get(f"{BASE_URL}/api/v1/eda/correlations", headers=headers).json()

def get_distribution(headers):
    return requests.get(f"{BASE_URL}/api/v1/eda/distribution", headers=headers).json()

def get_new_features(headers):
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/features/new-features", headers=headers).json()["novas_features"]
    )

def get_normalized(headers, method="minmax"):
    return pd.DataFrame(
        requests.get(f"{BASE_URL}/api/v1/features/normalize?method={method}", headers=headers).json()["amostra_normalizada"]
    )

def get_encoded(headers):
    return requests.get(f"{BASE_URL}/api/v1/features/encode", headers=headers).json()

def get_regression_simple(headers):
    return requests.get(f"{BASE_URL}/api/v1/regression/simple", headers=headers).json()

def get_regression_multiple(headers):
    return requests.get(f"{BASE_URL}/api/v1/regression/multiple", headers=headers).json()

def get_classification_knn(headers, k=5):
    return requests.get(f"{BASE_URL}/api/v1/classification/knn?k={k}", headers=headers).json()

def get_classification_rf(headers):
    return requests.get(f"{BASE_URL}/api/v1/classification/random-forest", headers=headers).json()

def get_kmeans(headers, n_clusters=3):
    return requests.get(f"{BASE_URL}/api/v1/clustering/kmeans?n_clusters={n_clusters}", headers=headers).json()

def get_outliers(headers, n_neighbors=20):
    return requests.get(f"{BASE_URL}/api/v1/clustering/outliers?n_neighbors={n_neighbors}", headers=headers).json()

def get_pca(headers, n_components=2):
    return requests.get(f"{BASE_URL}/api/v1/pca/run?n_components={n_components}", headers=headers).json()

def get_association(headers, min_support=0.1):
    return requests.get(f"{BASE_URL}/api/v1/pca/association?min_support={min_support}", headers=headers).json()