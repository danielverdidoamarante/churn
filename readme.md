# Projeto: Análise de Abandono de Clientes

Este projeto tem como objetivo realizar uma análise sobre os clientes de uma empresa, com foco em identificar fatores que influenciam o abandono (churn) de clientes. Para isso, usamos diversas técnicas de análise de dados e machine learning, incluindo a criação de gráficos e a aplicação de modelos preditivos.

## Estrutura do Repositório

**`data`**: 
   - Pasta com arquivos de dados com informações dos clientes.

**`EDA.ipynb`**: 
   - Análise exploratória dos dados, incluindo distribuição, estatísticas descritivas e visualizações de variáveis numéricas e categóricas.

**`preprocess.py`**: 
   - Realiza o pré-processamento dos dados antes do treinamento, incluindo a verificação de tipos de separadores no arquivo CSV e a transformação de variáveis categóricas.

**`train_models.py`**: 
   - Treina diferentes modelo de classificação para prever a variável `Exited` (abandonamento de clientes).

**`train.py`**: 
   - Treina um modelo de classificação usando o algoritmo Random Forest para prever a variável de destino `Exited` (abandonamento de clientes).
   - O modelo é salvo em um arquivo para ser utilizado posteriormente em previsões.

**`models`**:
   - Pasta com modelos treinados e salvos para gerar previsões.
   
**`predict.py`**: 
   - Carrega o modelo treinado e realiza previsões para novos dados.
   - Os resultados das previsões são salvos em um arquivo CSV, com duas colunas: `rowNumber` e `predictedValues`.
   - Uso: python predict.py <caminho_do_arquivo_csv>

**`output`**:
   - Pasta com arquivos gerados com o resultado das previsões.

## Dependências

Este projeto usa as seguintes bibliotecas e ferramentas:

- `pandas`: Para manipulação de dados.
- `scikit-learn`: Para modelos de machine learning (Random Forest, etc.).
- `matplotlib` e `seaborn`: Para visualização dos dados.
- `joblib`: Para salvar e carregar o modelo treinado.
  
Para instalar as dependências, utilize o seguinte comando:

```bash
pip install -r requirements.txt
```
## Como executar

Para gerar previsões em um novo arquivo, adicione o arquivo CSV à pasta `data` e execute o script predict.py, por exemplo:
```bash
python3 predict.py data/Abandono_teste.csv
```

Caso o separador seja diferente de ponto e vírgula (;), você pode especificar isso:
```bash
python3 predict.py data/Abandono_teste.csv ,
```