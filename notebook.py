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

# 1. CRÉATION DE DONNÉES SYNTHÉTIQUES RÉALISTES
print("\n📊 CRÉATION DU DATASET SYNTHÉTIQUE")
print("="*40)

np.random.seed(42)  # Pour la reproductibilité
n_samples = 5000

print(f"📋 Génération de {n_samples} échantillons...")

# Génération de données réalistes pour le crédit
data_list = []

for i in range(n_samples):
    # Revenus annuels (distribution log-normale)
    income = np.random.lognormal(mean=10.5, sigma=0.7)
    income = max(15000, min(income, 300000))  # Entre 15k et 300k
    
    # Montant du crédit demandé (2x à 8x les revenus)
    credit_multiplier = np.random.uniform(2, 8)
    credit_amount = income * credit_multiplier
    
    # Annuités mensuelles (5% à 20% des revenus annuels / 12)
    annuity_rate = np.random.uniform(0.05, 0.20)
    annuity = (income * annuity_rate) / 12
    
    # Âge (distribution normale centrée sur 40 ans)
    age = max(18, min(70, int(np.random.normal(40, 12))))
    
    # Années d'emploi (distribution exponentielle)
    employment_years = max(0, min(40, int(np.random.exponential(7))))
    
    # Ratios financiers calculés
    credit_income_ratio = credit_amount / income
    annuity_income_ratio = annuity / income
    
    # Variables famille
    family_size = max(1, min(6, int(np.random.poisson(2.3))))
    children_ratio = max(0, min(1, np.random.beta(2, 5)))
    
    # Score externe moyen (sources externes de crédit)
    external_sources_mean = np.random.beta(5, 5)
    
    # Variables catégorielles encodées
    education_encoded = np.random.randint(0, 5)
    income_type_encoded = np.random.randint(0, 4)
    family_status_encoded = np.random.randint(0, 4)
    code_gender = np.random.randint(0, 2)  # 0=F, 1=M
    
    # CALCUL DU RISQUE BASÉ SUR DES RÈGLES MÉTIER RÉALISTES
    risk_score = 0
    
    # Facteur 1: Ratio crédit/revenus
    if credit_income_ratio > 6:
        risk_score += 0.4
    elif credit_income_ratio > 4:
        risk_score += 0.2
    elif credit_income_ratio > 3:
        risk_score += 0.1
    
    # Facteur 2: Taux d'endettement
    if annuity_income_ratio > 0.35:
        risk_score += 0.3
    elif annuity_income_ratio > 0.25:
        risk_score += 0.15
    
    # Facteur 3: Âge
    if age < 25:
        risk_score += 0.15
    elif age > 65:
        risk_score += 0.1
    elif 30 <= age <= 50:
        risk_score -= 0.05  # Âge optimal
    
    # Facteur 4: Stabilité professionnelle
    if employment_years < 1:
        risk_score += 0.25
    elif employment_years < 3:
        risk_score += 0.1
    elif employment_years > 10:
        risk_score -= 0.1  # Bonus stabilité
    
    # Facteur 5: Niveau de revenus
    if income < 25000:
        risk_score += 0.15
    elif income > 75000:
        risk_score -= 0.05
    
    # Facteur 6: Situation familiale
    if family_size > 4:
        risk_score += 0.05
    
    # Ajout de variabilité aléatoire
    risk_score += np.random.normal(0, 0.1)
    
    # Décision binaire (TARGET: 1=défaut, 0=remboursement)
    # Seuil ajusté pour avoir ~8% de défaut comme dans la réalité
    target = 1 if risk_score > 0.6 else 0
    
    # Construction de la ligne de données
    row = [
        income,                 # AMT_INCOME_TOTAL
        credit_amount,          # AMT_CREDIT  
        annuity,                # AMT_ANNUITY
        age,                    # AGE_YEARS
        employment_years,       # EMPLOYMENT_YEARS
        credit_income_ratio,    # CREDIT_INCOME_RATIO
        annuity_income_ratio,   # ANNUITY_INCOME_RATIO
        family_size,            # FAMILY_SIZE
        children_ratio,         # CHILDREN_RATIO
        external_sources_mean,  # EXTERNAL_SOURCES_MEAN
        education_encoded,      # EDUCATION_ENCODED
        income_type_encoded,    # INCOME_TYPE_ENCODED
        family_status_encoded,  # FAMILY_STATUS_ENCODED
        code_gender,            # CODE_GENDER
        target                  # TARGET
    ]
    
    data_list.append(row)

# Conversion en DataFrame
feature_names = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AGE_YEARS', 'EMPLOYMENT_YEARS', 
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
    'FAMILY_SIZE', 'CHILDREN_RATIO',
    'EXTERNAL_SOURCES_MEAN',
    'EDUCATION_ENCODED', 'INCOME_TYPE_ENCODED', 
    'FAMILY_STATUS_ENCODED', 'CODE_GENDER'
]

columns = feature_names + ['TARGET']
df = pd.DataFrame(data_list, columns=columns)

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

# Fonction de test simple
def test_prediction(income, credit_amount, annuity, age, employment_years, gender='M'):
    """Test simple du modèle"""
    
    # Calcul des ratios
    credit_income_ratio = credit_amount / income
    annuity_income_ratio = annuity / income
    
    # Création du vecteur de features
    features_test = np.array([[
        income,                    # AMT_INCOME_TOTAL
        credit_amount,             # AMT_CREDIT
        annuity,                   # AMT_ANNUITY
        age,                       # AGE_YEARS
        employment_years,          # EMPLOYMENT_YEARS
        credit_income_ratio,       # CREDIT_INCOME_RATIO
        annuity_income_ratio,      # ANNUITY_INCOME_RATIO
        2,                         # FAMILY_SIZE (défaut)
        0.5,                       # CHILDREN_RATIO (défaut)
        0.5,                       # EXTERNAL_SOURCES_MEAN (défaut)
        1,                         # EDUCATION_ENCODED (défaut)
        1,                         # INCOME_TYPE_ENCODED (défaut)
        1,                         # FAMILY_STATUS_ENCODED (défaut)
        1 if gender == 'M' else 0  # CODE_GENDER
    ]])
    
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
