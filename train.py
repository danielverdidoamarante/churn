import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # Usado para salvar o modelo

def train_random_forest():
    # Carregar os dados
    df = pd.read_csv('data/Abandono_clientes.csv')
    
    # Separar features e target
    X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
    y = df['Exited']

    # Identificar variáveis categóricas e numéricas
    categorical_features = ['Gender', 'Geography']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    # Pré-processamento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Criar o modelo Random Forest com class_weight='balanced'
    random_forest = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    random_forest.fit(X_train, y_train)

    # Salvar o modelo treinado
    joblib.dump(random_forest, 'models/random_forest_model.pkl')

    print("Modelo Random Forest treinado e salvo com sucesso.")

if __name__ == "__main__":
    train_random_forest()
