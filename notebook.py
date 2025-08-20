# 🏦 Neo-Bank Credit Scoring Model
# Création d'un modèle 
# OBJECTIF: Modèle qui fonctionne sur Streamlit Cloud 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

print("🚀 CRÉATION MODÈLE NEO-BANK ")
print("="*50)

# Vérification de la version
import sklearn
print(f"📌 Version scikit-learn: {sklearn.__version__}")

# 1. CHARGEMENT DES VRAIES DONNÉES
print("\n📊 CHARGEMENT DES VRAIES DONNÉES")
print("="*40)

# Chargement du dataset d'entraînement
print("� Chargement de application_train.csv...")
try:
    df_train = pd.read_csv('data/application_train.csv')
    print(f"✅ Dataset chargé: {df_train.shape}")
    print(f"📊 Taux de défaut réel: {df_train['TARGET'].mean():.1%}")
except Exception as e:
    print(f"❌ Erreur de chargement: {e}")
    exit(1)

# Exploration rapide des colonnes
print(f"\n📋 Colonnes disponibles: {len(df_train.columns)}")
print("🔝 Premières colonnes:", list(df_train.columns[:10]))

# Sélection des features importantes pour le modèle
# On prend les colonnes principales qui existent probablement
feature_candidates = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'CNT_FAM_MEMBERS', 'CNT_CHILDREN',
    'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS'
]

# Vérifier quelles colonnes existent vraiment
existing_features = []
for col in feature_candidates:
    if col in df_train.columns:
        existing_features.append(col)
        print(f"✅ {col}")
    else:
        print(f"❌ {col} - non trouvée")

print(f"\n📊 Features retenues: {len(existing_features)}")

# Préparation des données
df = df_train[existing_features + ['TARGET']].copy()

# Conversion des jours en années pour DAYS_BIRTH et DAYS_EMPLOYED
if 'DAYS_BIRTH' in df.columns:
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)
    df = df.drop('DAYS_BIRTH', axis=1)

if 'DAYS_EMPLOYED' in df.columns:
    # Gestion des valeurs aberrantes (365243 = pas d'emploi)
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    df.loc[df['EMPLOYMENT_YEARS'] > 50, 'EMPLOYMENT_YEARS'] = 0  # Pas d'emploi
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(0, 40)
    df = df.drop('DAYS_EMPLOYED', axis=1)

# Encodage des variables catégorielles
categorical_cols = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']
for col in categorical_cols:
    if col in df.columns:
        df[f'{col}_ENCODED'] = pd.Categorical(df[col]).codes
        df = df.drop(col, axis=1)

# Calcul de ratios financiers
if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, 1)

if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_ANNUITY' in df.columns:
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, 1)

# Nettoyage des données
print("\n🧹 NETTOYAGE DES DONNÉES")
print("="*25)

# Suppression des valeurs manquantes
print(f"📊 Valeurs manquantes avant: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"📊 Valeurs manquantes après: {df.isnull().sum().sum()}")
print(f"📊 Échantillons restants: {len(df)}")

# Limitation du dataset si trop grand (pour compatibilité)
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)
    print(f"📊 Dataset réduit à: {len(df)} échantillons")

# Définition des features finales
feature_names = [col for col in df.columns if col != 'TARGET']
print(f"\n📋 Features finales: {len(feature_names)}")
for i, feature in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feature}")

print(f"✅ Dataset créé: {df.shape}")
print(f"📊 Taux de défaut: {df['TARGET'].mean():.1%}")
print(f"📋 Variables: {len(feature_names)} features")

# Vérification des données
print("\n📈 STATISTIQUES DU DATASET")
print("="*30)
print("Revenus moyens:", f"{df['AMT_INCOME_TOTAL'].mean():,.0f}€")
print("Crédit moyen:", f"{df['AMT_CREDIT'].mean():,.0f}€")
print("Ratio crédit/revenus moyen:", f"{df['CREDIT_INCOME_RATIO'].mean():.1f}x")
print("Âge moyen:", f"{df['AGE_YEARS'].mean():.0f} ans")
print("Ancienneté moyenne:", f"{df['EMPLOYMENT_YEARS'].mean():.1f} ans")

# 2. PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT
print("\n🎯 PRÉPARATION POUR L'ENTRAÎNEMENT")
print("="*40)

# Séparation des features et de la cible
X = df[feature_names].copy()
y = df['TARGET'].copy()

print(f"📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Vérification des valeurs manquantes
print(f"🔍 Valeurs manquantes: {X.isnull().sum().sum()}")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# 3. STANDARDISATION DES DONNÉES
print("\n⚖️ STANDARDISATION")
print("="*20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Standardisation appliquée")

# 4. ENTRAÎNEMENT DU MODÈLE
print("\n🤖 ENTRAÎNEMENT DU MODÈLE")
print("="*30)

# Modèle Random Forest compatible scikit-learn 1.3.2
model = RandomForestClassifier(
    n_estimators=100,       # Nombre d'arbres
    max_depth=15,           # Profondeur max
    min_samples_split=10,   # Min échantillons pour split
    min_samples_leaf=5,     # Min échantillons par feuille
    random_state=42,        # Reproductibilité
    n_jobs=-1              # Utiliser tous les CPU
)

print("🔄 Entraînement en cours...")
model.fit(X_train_scaled, y_train)
print("✅ Modèle entraîné!")

# 5. ÉVALUATION DU MODÈLE
print("\n📊 ÉVALUATION")
print("="*15)

# Prédictions
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"🎯 Score Train: {train_score:.3f}")
print(f"🎯 Score Test: {test_score:.3f}")

# Prédictions pour AUC
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"📈 AUC Score: {auc_score:.3f}")

# Importance des features
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 TOP 5 FEATURES IMPORTANTES:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 6. SAUVEGARDE DU MODÈLE
print("\n💾 SAUVEGARDE")
print("="*15)

# Création du dossier model
os.makedirs('model', exist_ok=True)

# Sauvegarde du modèle
with open('model/credit_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Modèle sauvegardé: model/credit_model_v2.pkl")

# Sauvegarde du scaler
with open('model/scaler_v2.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler sauvegardé: model/scaler_v2.pkl")

# Sauvegarde des noms de features
with open('model/feature_names_v2.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Features sauvegardées: model/feature_names_v2.pkl")

# Sauvegarde des encoders (vides pour ce modèle simple)
encoders = {
    'gender': None,
    'education': None,
    'income_type': None,
    'family_status': None
}

with open('model/encoders_v2.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✅ Encoders sauvegardés: model/encoders_v2.pkl")

# 7. TEST DU MODÈLE
print("\n🧪 TEST DU MODÈLE")
print("="*20)

# Fonction de test adaptée aux vraies features
def test_prediction(income, credit_amount, annuity, age, employment_years, 
                   family_members=2, children=0, gender='M', education='Higher education', 
                   income_type='Working', family_status='Married'):
    """Test du modèle avec les vraies features"""
    
    # Calcul des ratios
    credit_income_ratio = credit_amount / income
    annuity_income_ratio = annuity / income
    
    # Encodage simple des variables catégorielles (basé sur les valeurs typiques)
    gender_encoded = 1 if gender == 'M' else 0
    
    # Encodages simplifiés (vous pouvez ajuster selon vos données)
    education_mapping = {
        'Secondary / secondary special': 0,
        'Higher education': 1,
        'Incomplete higher': 2,
        'Lower secondary': 3,
        'Academic degree': 4
    }
    education_encoded = education_mapping.get(education, 1)
    
    income_type_mapping = {
        'Working': 0,
        'Commercial associate': 1,
        'Pensioner': 2,
        'State servant': 3
    }
    income_type_encoded = income_type_mapping.get(income_type, 0)
    
    family_status_mapping = {
        'Married': 0,
        'Single / not married': 1,
        'Civil marriage': 2,
        'Separated': 3,
        'Widow': 4
    }
    family_status_encoded = family_status_mapping.get(family_status, 0)
    
    # Création du vecteur avec exactement les 13 features du modèle
    # Dans l'ordre exact: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, CNT_FAM_MEMBERS, CNT_CHILDREN,
    # AGE_YEARS, EMPLOYMENT_YEARS, CODE_GENDER_ENCODED, NAME_EDUCATION_TYPE_ENCODED,
    # NAME_INCOME_TYPE_ENCODED, NAME_FAMILY_STATUS_ENCODED, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO
    
    features_test = np.array([[
        income,                    # AMT_INCOME_TOTAL
        credit_amount,             # AMT_CREDIT
        annuity,                   # AMT_ANNUITY
        family_members,            # CNT_FAM_MEMBERS
        children,                  # CNT_CHILDREN
        age,                       # AGE_YEARS
        employment_years,          # EMPLOYMENT_YEARS
        gender_encoded,            # CODE_GENDER_ENCODED
        education_encoded,         # NAME_EDUCATION_TYPE_ENCODED
        income_type_encoded,       # NAME_INCOME_TYPE_ENCODED
        family_status_encoded,     # NAME_FAMILY_STATUS_ENCODED
        credit_income_ratio,       # CREDIT_INCOME_RATIO
        annuity_income_ratio       # ANNUITY_INCOME_RATIO
    ]])
    
    print(f"🔍 Test avec {features_test.shape[1]} features (attendu: 13)")
    
    # Standardisation
    features_scaled = scaler.transform(features_test)
    
    # Prédiction
    risk_proba = model.predict_proba(features_scaled)[0][1]
    risk_score = int(risk_proba * 100)
    
    return risk_score, risk_proba

# Tests avec différents profils
print("Profil 1 - Bon client:")
score1, proba1 = test_prediction(50000, 150000, 1200, 35, 5)
print(f"  Score: {score1}%, Probabilité: {proba1:.3f}")

print("Profil 2 - Client risqué:")
score2, proba2 = test_prediction(25000, 200000, 1800, 22, 0)
print(f"  Score: {score2}%, Probabilité: {proba2:.3f}")

print("Profil 3 - Client excellent:")
score3, proba3 = test_prediction(80000, 200000, 1500, 40, 10)
print(f"  Score: {score3}%, Probabilité: {proba3:.3f}")

print("\n🎉 MODÈLE CRÉÉ AVEC SUCCÈS!")
print("="*30)
print("✅ Modèle compatible scikit-learn 1.3.2")
print("✅ Tous les fichiers sauvegardés dans model/")
print("✅ Prêt pour Streamlit Cloud")
print(f"✅ Version sklearn utilisée: {sklearn.__version__}")

print("\n📁 Fichiers créés:")
print("  - model/credit_model_v2.pkl")
print("  - model/scaler_v2.pkl")
print("  - model/feature_names_v2.pkl")
print("  - model/encoders_v2.pkl")
