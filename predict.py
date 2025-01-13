import pandas as pd
import joblib
import os
import sys

def predict(input_file, separator = ';'):
    # Carregar o modelo treinado
    model = joblib.load('random_forest_model.pkl')

    # Carregar os dados de entrada
    df = pd.read_csv(input_file, sep=separator)

    # Separar as features (presumindo que as mesmas colunas sejam usadas)
    X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # Fazer as predições
    predictions = model.predict(X)

    # Criar o DataFrame com rowNumber e predictedValues
    result = pd.DataFrame({
        'rowNumber': df['RowNumber'],
        'predictedValues': predictions
    })

    # Criar a pasta 'output' se não existir
    os.makedirs('output', exist_ok=True)

    # Salvar o resultado final em um arquivo CSV dentro da pasta 'output'
    result.to_csv('output/resultado_churn.csv', index=False)

    print("Predições salvas em 'output/resultado_churn.csv'.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        predict(input_file)
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        sep = sys.argv[2]
        predict(input_file, sep)
    else:
        print("Uso: python predict.py <caminho_do_arquivo_csv> [opcional: separador (por exemplo , ou ;)]")