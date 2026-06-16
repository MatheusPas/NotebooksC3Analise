# Divisão de Tarefas — Hackathon Data Analysis & ML
**Projeto:** Analisando Dados de Preços de Casas nos Estados Unidos  
**Professor:** Howard Roatti — FAESA Centro Universitário  
**Entrega:** GitHub (link via AVA) | Apresentação: Jupyter Notebook (Story Telling)

---

## Equipe e Responsabilidades

---

### Matheus
**Responsabilidade:** EDA + Feature Engineering + Visualização de Dados + Manutenção da Infra

#### Tarefa 1 — Análise Exploratória de Dados + Feature Engineering (1 ponto)
Realizar a análise inicial do dataset identificando:
- Quais variáveis estão presentes no dataset
- Quais são numéricas e quais são categóricas
- Distribuição das variáveis (histogramas, boxplots)
- Correlações entre variáveis (heatmap de correlação)
- Identificação de valores faltantes (missing values)
- Identificação inicial de outliers

Com base na análise, realizar transformações no dataset:
- Tratar valores faltantes (imputação por média, mediana ou moda)
- Normalização/padronização das variáveis numéricas (MinMaxScaler ou StandardScaler)
- Codificação das variáveis categóricas (One-Hot Encoding ou Label Encoding)
- Criar novas features que possam melhorar os modelos, como:
  - `house_age` = ano atual − `year_built`
  - `remod_age` = ano atual − `year_remod_add`
  - `total_bath` = `full_bath` + (0.5 × `half_bath`)
  - `area_per_room` = `gr_liv_area` / `totrms_abvgrd`
- Selecionar as variáveis mais relevantes para os modelos

#### Tarefa 7 — Visualização de Dados (1 ponto)
Criar visualizações para apresentação dos resultados:
- Gráficos de distribuição do preço de venda
- Heatmap de correlação entre variáveis numéricas
- Gráficos de dispersão entre as variáveis mais relevantes e o preço
- Visualizações dos resultados de cada modelo (gráficos de erro, comparação)
- Gráficos do Story Telling final

#### Responsabilidade Extra
- Manutenção da VM, PostgreSQL e API durante o projeto
- Organização e entrega do repositório no GitHub (0,5 ponto)

**Alternativas de encoding:**
- One-Hot Encoding ← recomendado para variáveis nominais
- Label Encoding ← para variáveis ordinais
- Target Encoding ← opcional, para variáveis de alta cardinalidade

---

### Gessiele
**Responsabilidade:** Aprendizagem Supervisionada — Classificação

#### Tarefa 3b — Classificação (1 ponto)
Converter o problema em classificação binária:
- Criar variável binária `price_category`:
  - `1` = preço alto (acima da mediana)
  - `0` = preço baixo (abaixo da mediana)
- Treinar um modelo de classificação
- Avaliar com métricas:
  - **Accuracy** — acurácia geral
  - **Precision** — precisão
  - **Recall** — revocação
  - **F1-Score** — média harmônica
- Plotar a matriz de confusão

**Alternativas aceitas pelo professor:**
- KNN (K-Nearest Neighbors) ← recomendado
- Naive Bayes
- Regressão Logística
- Árvore de Decisão
- Random Forest

---

### Eduardo
**Responsabilidade:** Aprendizagem Supervisionada — Regressão

#### Tarefa 3a — Regressão Linear (1 ponto)
Criar modelos de regressão para prever o preço de venda:
- Dividir o dataset em treino e teste (80/20 ou 70/30)
- Treinar um modelo de Regressão Linear simples (1 variável) com `gr_liv_area`
- Treinar um modelo de Regressão Linear múltipla (várias variáveis)
- Avaliar os modelos com as métricas:
  - **RMSE** — Root Mean Squared Error
  - **MAE** — Mean Absolute Error
  - **R²** — Coeficiente de Determinação
- Plotar os valores reais vs. preditos
- Interpretar os resultados

**Alternativas aceitas pelo professor:**
- Regressão Linear ← recomendado
- Random Forest Regressor
- XGBoost Regressor ← modelo mais avançado, opcional

---

### Lucas
**Responsabilidade:** Redução de Dimensionalidade + Organização do GitHub

#### Tarefa 4b — Redução de Dimensionalidade (1 ponto)
Aplicar PCA (Principal Component Analysis) ao dataset:
- Padronizar os dados antes de aplicar o PCA
- Determinar o número ideal de componentes principais
- Plotar a variância explicada acumulada (elbow/scree plot)
- Visualizar os dados em 2D após redução
- Interpretar quais variáveis mais contribuem para cada componente

**Alternativas aceitas pelo professor:**
- PCA (Principal Component Analysis) ← recomendado
- t-SNE (para visualização em 2D/3D)
- LDA (Linear Discriminant Analysis)

#### Responsabilidade Extra
- Configurar o repositório GitHub com estrutura organizada
- Conectar os notebooks Jupyter à API
- Revisar e montar o notebook final para apresentação
- Organização do repositório (0,5 ponto)

---

### João
**Responsabilidade:** Clusterização + Análise de Associação + Análise de Outlier

#### Tarefa 4a — Clusterização (1 ponto)
Identificar grupos de casas com características semelhantes:
- Aplicar K-Means ao dataset (após PCA ou nas features principais)
- Usar o método Elbow para definir o número ideal de clusters
- Visualizar os clusters em 2D
- Interpretar o perfil de cada cluster (casas baratas, médias, caras)

**Alternativas aceitas pelo professor:**
- K-Means ← recomendado
- DBSCAN
- Hierarchical Clustering

#### Tarefa 4c e 4d — Associação + Outlier (1 ponto)
**Análise de Associação:**
- Discretizar variáveis contínuas em categorias (ex: qualidade alta/média/baixa)
- Aplicar o algoritmo **Apriori** para encontrar regras de associação
- Interpretar as regras encontradas (ex: "casas com qualidade alta E grande área → preço alto")
- Avaliar com métricas de **suporte**, **confiança** e **lift**

**Detecção de Outliers:**
- Aplicar o algoritmo **Local Outlier Factor (LOF)**
- Identificar as casas consideradas outliers
- Visualizar os outliers no dataset
- Interpretar por que essas casas são atípicas

---

## Conexão com a Infraestrutura

Todos os membros se conectam ao servidor via Radmin VPN:

```python
import requests
import pandas as pd

# Login
response = requests.post("http://26.246.114.225:8000/auth/login", json={
    "username": "seu_usuario",
    "password": "sua_senha"
})
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Carregar dataset completo
df = pd.DataFrame(
    requests.get("http://26.246.114.225:8000/api/v1/houses?limit=1460", headers=headers).json()
)
```

---

## Estrutura dos Repositórios

**Repositório da API:** https://github.com/MatheusPas/TrabalhoC3Analise
```
TrabalhoC3Analise/
├── README.md
├── docs/
│   ├── api-documentation.md
│   └── divisao-tarefas.md
├── data/
│   └── train.csv
└── api/
```

**Repositório dos Notebooks:** https://github.com/MatheusPas/NotebooksC3Analise
```
NotebooksC3Analise/
├── .gitignore
├── Connection.py
├── ParteMatheus.ipynb              (Matheus - EDA & Feature Engineering)
├── ParteLucas.ipynb                (Lucas - Redução de Dimensionalidade / PCA)
├── ParteEduardo.ipynb              (Eduardo - Regressão Linear Simples & Múltipla)
└── README.md
```