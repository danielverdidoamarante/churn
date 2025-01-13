import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import preprocess_data

def train_model():
    # Carregar e pré-processar os dados
    X, y, preprocessor = preprocess_data('data/Abandono_clientes.csv')
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelos
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    random_forest = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Geography'])], 
            remainder='passthrough')),  # Random Forest não precisa de escalonamento
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    gradient_boosting = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Treinar e avaliar Logistic Regression
    logistic_regression.fit(X_train, y_train)
    lr_preds = logistic_regression.predict(X_test)
    lr_probs = logistic_regression.predict_proba(X_test)[:, 1]
    print("Logistic Regression Metrics:")
    print(classification_report(y_test, lr_preds))
    print(f"AUC-ROC: {roc_auc_score(y_test, lr_probs):.2f}")
    
    # Treinar e avaliar Random Forest
    random_forest.fit(X_train, y_train)
    rf_preds = random_forest.predict(X_test)
    rf_probs = random_forest.predict_proba(X_test)[:, 1]
    print("\nRandom Forest Metrics:")
    print(classification_report(y_test, rf_preds))
    print(f"AUC-ROC: {roc_auc_score(y_test, rf_probs):.2f}")

    # Treinar e avaliar Gradient Boosting
    gradient_boosting.fit(X_train, y_train)
    gb_preds = gradient_boosting.predict(X_test)
    gb_probs = gradient_boosting.predict_proba(X_test)[:, 1]
    print("\nGradient Boosting Metrics:")
    print(classification_report(y_test, gb_preds))
    print(f"AUC-ROC: {roc_auc_score(y_test, gb_probs):.2f}")
    
    return random_forest

if __name__ == "__main__":
    train_model()