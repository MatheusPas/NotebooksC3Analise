# 📁 /notebooks

Esta pasta contém os Jupyter Notebooks de cada membro da equipe, organizados por tarefa.

| # | Notebook | Responsável | Tarefa |
|---|----------|-------------|--------|
| 01 | `01_eda.ipynb` | Matheus | Análise Exploratória de Dados |
| 02 | `02_feature_engineering.ipynb` | Matheus | Feature Engineering |
| 03 | `03_regressao.ipynb` | Eduardo | Regressão Linear |
| 04 | `04_classificacao.ipynb` | Gessiele | Classificação (KNN / RF) |
| 05 | `05_clusterizacao.ipynb` | João | Clusterização (K-Means) |
| 06 | `06_dimensionalidade.ipynb` | Lucas | Redução de Dimensionalidade (PCA) |
| 07 | `07_associacao_outlier.ipynb` | João | Associação + Detecção de Outliers |
| 08 | `08_storytelling_final.ipynb` | Todos | Apresentação Final (Story Telling) |

## Como usar

1. Conecte-se à VPN (Radmin VPN)
2. Certifique-se de que o arquivo `Connection.py` está acessível
3. Execute as células do notebook em ordem

```python
import Connection as conn
my_conn = conn.get_conn("seu_usuario", "sua_senha")
df = conn.puxar_df(my_conn)
```
