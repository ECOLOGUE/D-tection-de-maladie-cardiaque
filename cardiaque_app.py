import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------
# Titre de l'application
st.title("Prédiction de Maladie Cardiaque")

st.image("Design-sans-titre.jpg", caption="Légende de l'image")
# -----------------------------
# Charger le dataset (optionnel pour info)
df = pd.read_csv(r'Cardique.csv', sep=';')
st.write("Aperçu du dataset :", df.head())

# -----------------------------
# Préparation des données
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# -----------------------------
# Entraîner le modèle (XGBoost)
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Affichage des performances
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

st.write("**Accuracy :**", round(accuracy_score(y_test, y_pred), 2))
st.write("**AUC :**", round(roc_auc_score(y_test, y_proba), 2))

# -----------------------------
# Section pour faire des prédictions personnalisées
st.header("Prédiction pour un nouveau patient")
st.markdown("""
### Correspondance des variables

- **Thallium** : Test au thallium  
- **Number of vessels fluro** : Nombre de vaisseaux (fluoroscopie)  
- **Exercise angina** : Angine d’effort  
- **Max HR** : Fréquence cardiaque maximale  
- **ST depression** : Dépression du segment ST  
- **Chest pain type** : Type de douleur thoracique
""")



# Créer des champs pour chaque variable du dataset
input_data = {}
for col in X.columns:
    # Pour les colonnes numériques
    input_data[col] = st.number_input(f"Valeur pour {col}", value= 0)

# Convertir en DataFrame pour la prédiction
input_df = pd.DataFrame([input_data])

# Bouton de prédiction
if st.button("Prédire"):
    
    # Vérifier si toutes les valeurs sont remplies
    if input_df.isnull().values.any():
        st.error("Veuillez entrer toutes les valeurs.")
    else:
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        st.success("Prédiction effectuée")
        st.write("Résultat :", "Risque de maladie cardiaque" if prediction[0] == 1 else "Pas de risque de maladie cardiaque")
        st.write("Probabilité :", round(proba * 100, 2), "%")

