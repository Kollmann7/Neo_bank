# Étape 1: Récupération et analyse du notebook Kaggle
# Applied Predictive Modelling (Brief Overview) by moizzz

import kaggle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import pickle
import warnings
warnings.filterwarnings('ignore')

# Vous avez déjà fait l'authentification Kaggle
# kaggle.api.authenticate()

print("🔍 ÉTAPE 1: ANALYSE DU NOTEBOOK KAGGLE")
print("="*50)



# 2. Chargement des données principales
print("\n📊 Chargement des données...")
# Fichier principal - informations sur les demandes de prêt
df_train = pd.read_csv('./data/application_train.csv')
df_test = pd.read_csv('./data/application_test.csv')

print(f"📋 Données d'entraînement: {df_train.shape}")
print(f"📋 Données de test: {df_test.shape}")

# Affichage des premières informations
print("\n🎯 Variable cible (TARGET):")
print(df_train['TARGET'].value_counts())
print(f"Taux de défaut: {df_train['TARGET'].mean():.2%}")

# 3. Analyse exploratoire adaptée au contexte néo-banque
print("\n🏦 ANALYSE POUR LA NÉO-BANQUE")
print("="*40)

# Variables importantes pour l'éligibilité au prêt
key_features = [
    'AMT_INCOME_TOTAL', 
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AGE_YEARS',
    'EMPLOYMENT_YEARS', 
    'CREDIT_INCOME_RATIO',
    'ANNUITY_INCOME_RATIO',
    'FAMILY_SIZE',
    'CHILDREN_RATIO',
    'EXTERNAL_SOURCES_MEAN',
    'EDUCATION_ENCODED',
    'INCOME_TYPE_ENCODED',
    'FAMILY_STATUS_ENCODED',
    'CODE_GENDER'
]

# Correction des DAYS_EMPLOYED aberrants
df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].replace(365243, np.nan)

# Création des variables dérivées (corrigées)
df_train['AGE_YEARS'] = (-df_train['DAYS_BIRTH'] / 365).astype(int)
df_train['EMPLOYMENT_YEARS'] = np.where(
    df_train['DAYS_EMPLOYED'].isna(), 
    0,  # Pas d'emploi déclaré
    (-df_train['DAYS_EMPLOYED'] / 365).clip(0, 50)  # Limiter à 50 ans max
).astype(int)

# Ratios d'endettement
df_train['CREDIT_INCOME_RATIO'] = df_train['AMT_CREDIT'] / df_train['AMT_INCOME_TOTAL']
df_train['ANNUITY_INCOME_RATIO'] = df_train['AMT_ANNUITY'] / df_train['AMT_INCOME_TOTAL']

# 2. AJOUT DE VARIABLES MÉTIER IMPORTANTES
print("📊 Ajout de variables métier...")

# Variables famille et logement
df_train['FAMILY_SIZE'] = df_train['CNT_FAM_MEMBERS']
df_train['CHILDREN_RATIO'] = df_train['CNT_CHILDREN'] / df_train['CNT_FAM_MEMBERS']

# Variables revenus externes
df_train['EXTERNAL_SOURCES_MEAN'] = df_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

# Encodage des variables catégorielles principales
le_education = LabelEncoder()
le_income_type = LabelEncoder()
le_family_status = LabelEncoder()

df_train['EDUCATION_ENCODED'] = le_education.fit_transform(df_train['NAME_EDUCATION_TYPE'].fillna('Unknown'))
df_train['INCOME_TYPE_ENCODED'] = le_income_type.fit_transform(df_train['NAME_INCOME_TYPE'].fillna('Unknown'))
df_train['FAMILY_STATUS_ENCODED'] = le_family_status.fit_transform(df_train['NAME_FAMILY_STATUS'].fillna('Unknown'))

# 3. SÉLECTION DES FEATURES AMÉLIORÉES
features_v2 = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AGE_YEARS', 'EMPLOYMENT_YEARS', 
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
    'FAMILY_SIZE', 'CHILDREN_RATIO',
    'EXTERNAL_SOURCES_MEAN',
    'EDUCATION_ENCODED', 'INCOME_TYPE_ENCODED', 'FAMILY_STATUS_ENCODED',
    'CODE_GENDER'
]

# Préparation du dataset
X_features = df_train[features_v2].copy()

# Encodage genre
le_gender = LabelEncoder()
X_features['CODE_GENDER'] = le_gender.fit_transform(X_features['CODE_GENDER'])

# Gestion des valeurs manquantes
X_features = X_features.fillna(X_features.median())

# Target
y = df_train['TARGET']

# Alignement des indices
X_features = X_features.loc[y.index]

print(f"✅ Dataset amélioré: {X_features.shape}")
print(f"📊 Variables: {list(X_features.columns)}")

# 4. ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ
print("\n🎯 ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ")
print("="*40)

# Division
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation
scaler_v2 = StandardScaler()
X_train_scaled = scaler_v2.fit_transform(X_train)
X_test_scaled = scaler_v2.transform(X_test)

# Modèle Random Forest optimisé
rf_v2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_v2.fit(X_train_scaled, y_train)

# Évaluation
y_pred_proba_v2 = rf_v2.predict_proba(X_test_scaled)[:, 1]
auc_v2 = roc_auc_score(y_test, y_pred_proba_v2)

print(f"🎯 AUC Score V2: {auc_v2:.3f}")

# Importance des features
feature_importance_v2 = pd.DataFrame({
    'feature': X_features.columns,
    'importance': rf_v2.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 TOP 5 VARIABLES IMPORTANTES:")
for i, row in feature_importance_v2.head(5).iterrows():
    print(f"  • {row['feature']}: {row['importance']:.3f}")

# 5. SAUVEGARDE DU MODÈLE AMÉLIORÉ
print("\n💾 SAUVEGARDE DU MODÈLE V2")
print("="*30)

# Création du dossier models s'il n'existe pas
import os
os.makedirs('./models', exist_ok=True)

# Sauvegarde
with open('./models/credit_model_v2.pkl', 'wb') as f:
    pickle.dump(rf_v2, f)

with open('./models/scaler_v2.pkl', 'wb') as f:
    pickle.dump(scaler_v2, f)

with open('./models/feature_names_v2.pkl', 'wb') as f:
    pickle.dump(list(X_features.columns), f)

# Sauvegarde des encoders
encoders = {
    'gender': le_gender,
    'education': le_education,
    'income_type': le_income_type,
    'family_status': le_family_status
}

with open('./models/encoders_v2.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Modèle V2 sauvegardé!")

# 6. FONCTION DE SCORING AMÉLIORÉE
def predict_default_risk_v2(client_data):
    """
    Fonction de scoring améliorée pour l'API
    
    Args:
        client_data (dict): Données du client
    
    Returns:
        dict: Score et recommandations
    """
    
    # Chargement du modèle
    with open('./models/credit_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('./models/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Calcul des features
    features = np.array([[
        client_data['income'],
        client_data['credit_amount'],
        client_data['annuity'],
        client_data['age'],
        client_data['employment_years'],
        client_data['credit_amount'] / client_data['income'],  # CREDIT_INCOME_RATIO
        client_data['annuity'] / client_data['income'],        # ANNUITY_INCOME_RATIO
        client_data.get('family_size', 2),
        client_data.get('children_ratio', 0),
        client_data.get('external_sources_mean', 0.5),
        client_data.get('education_encoded', 1),
        client_data.get('income_type_encoded', 1),
        client_data.get('family_status_encoded', 1),
        1 if client_data['gender'] == 'M' else 0
    ]])
    
    # Prédiction
    features_scaled = scaler.transform(features)
    risk_proba = model.predict_proba(features_scaled)[0][1]
    
    # Interprétation
    risk_score = int(risk_proba * 100)
    
    if risk_score < 15:
        risk_level = "TRÈS FAIBLE"
        recommendation = "APPROUVER"
        explanation = "Excellent profil client, risque minimal"
    elif risk_score < 30:
        risk_level = "FAIBLE"
        recommendation = "APPROUVER"
        explanation = "Bon profil client, risque acceptable"
    elif risk_score < 50:
        risk_level = "MODÉRÉ"
        recommendation = "APPROUVER avec conditions"
        explanation = "Profil correct, surveiller l'évolution"
    elif risk_score < 70:
        risk_level = "ÉLEVÉ"
        recommendation = "ÉTUDIER"
        explanation = "Profil à risque, analyse approfondie requise"
    else:
        risk_level = "TRÈS ÉLEVÉ"
        recommendation = "REFUSER"
        explanation = "Risque de défaut trop important"
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'explanation': explanation,
        'model_version': 'v2',
        'details': {
            'credit_income_ratio': f"{(client_data['credit_amount'] / client_data['income']):.2%}",
            'annuity_income_ratio': f"{(client_data['annuity'] / client_data['income']):.2%}",
            'employment_years': client_data['employment_years'],
            'family_size': client_data.get('family_size', 2)
        }
    }

# Test de la fonction
print("\n🧪 TEST DE LA FONCTION DE SCORING V2")
print("="*35)

test_client = {
    'income': 50000,
    'credit_amount': 200000,
    'annuity': 1500,
    'age': 35,
    'employment_years': 5,
    'gender': 'M',
    'family_size': 2,
    'children_ratio': 0.5,
    'external_sources_mean': 0.6,
    'education_encoded': 2,
    'income_type_encoded': 1,
    'family_status_encoded': 1
}

test_result = predict_default_risk_v2(test_client)

print("Test avec un profil client:")
for key, value in test_result.items():
    print(f"  {key}: {value}") 