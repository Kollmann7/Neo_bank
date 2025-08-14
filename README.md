# 🏦 Neo-Bank Credit Dashboard

**Tableau de bord de scoring crédit en temps réel pour conseillers clientèle**

## 🚀 Déployé sur Streamlit Cloud

### 📊 Fonctionnalités

- **Évaluation instantanée** de risque de crédit
- **Interface intuitive** pour conseillers bancaires  
- **Modèle ML intégré** (Random Forest optimisé)
- **Conformité RGPD** complète
- **Déploiement direct** depuis GitHub

### 🔧 Technologies

- **Frontend :** Streamlit
- **Backend :** Moteur de scoring intégré (pas d'API externe)
- **ML :** Scikit-learn, Random Forest
- **Déploiement :** Streamlit Cloud + GitHub
- **RGPD :** Consentement, minimisation, anonymisation

### 🎯 Architecture Simplifiée

```
GitHub Repository
    ↓
Streamlit Cloud (auto-deploy)
    ↓
streamlit_app.py (point d'entrée)
    ↓
dashboard/app.py (interface principale)
    ↓
components/
    ├── credit_scoring_engine.py (moteur ML intégré)
    ├── api_client.py (interface locale)
    ├── client_form.py (formulaire client)
    ├── risk_interpretation.py (visualisations)
    └── rgpd_compliance.py (conformité RGPD)
    ↓
model/credit_model_v2.pkl (modèle ML)
```

### 🚀 Déploiement Streamlit Cloud

1. **Push sur GitHub**
   ```bash
   git add .
   git commit -m "Deploy Neo-Bank Dashboard"
   git push origin main
   ```

2. **Configuration Streamlit Cloud :**
   - Repository : `votre-username/Neo_bank`
   - Branch : `main`
   - Main file : `streamlit_app.py`

3. **Variables d'environnement** (si nécessaire) :
   ```
   PYTHONPATH=/app
   ```

### 📋 Structure des fichiers

```
Neo_bank/
├── streamlit_app.py          # Point d'entrée Streamlit Cloud
├── notebook.py               # Script de création du modèle
├── model/
│   └── credit_model_v2.pkl   # Modèle ML entraîné
├── dashboard/
│   ├── app.py                # Interface principale
│   ├── requirements.txt      # Dépendances Python
│   └── components/
│       ├── credit_scoring_engine.py  # Moteur ML intégré
│       ├── api_client.py             # Client API local
│       ├── client_form.py            # Formulaire client
│       ├── risk_interpretation.py    # Visualisations
│       └── rgpd_compliance.py        # Conformité RGPD
└── README.md
```

### 🔒 Conformité RGPD

- ✅ **Consentement explicite** avant traitement
- ✅ **Aucun stockage** des données personnelles
- ✅ **Minimisation** automatique des données
- ✅ **Logs anonymisés** uniquement
- ✅ **Traitement temps réel** sans persistance
- ✅ **Notice de confidentialité** intégrée

### 🎯 Utilisation

1. **Consentement RGPD** : Accepter le traitement des données
2. **Saisie client** : Remplir les informations nécessaires
3. **Évaluation** : Clic sur "Évaluer le Client"  
4. **Résultats** : Score, recommandation et visualisations
5. **Décision** : Approuver, étudier ou refuser le dossier

### 📊 Scoring & Interprétation

| Score | Niveau | Recommandation | Action |
|-------|---------|----------------|---------|
| 0-15% | TRÈS FAIBLE | APPROUVER | ✅ Financement immédiat |
| 15-30% | FAIBLE | APPROUVER | ✅ Bon dossier |
| 30-50% | MODÉRÉ | APPROUVER + suivi | ⚠️ Surveillance |
| 50-70% | ÉLEVÉ | ÉTUDIER | 🔍 Analyse approfondie |
| 70%+ | TRÈS ÉLEVÉ | REFUSER | ❌ Risque critique |

### 🛠️ Développement local

```bash
# Cloner le repository
git clone https://github.com/votre-username/Neo_bank.git
cd Neo_bank

# Installer les dépendances
pip install -r dashboard/requirements.txt

# Lancer l'application
streamlit run streamlit_app.py
```

### 📞 Support

- **Email :** privacy@neo-bank.fr
- **Documentation :** README.md
- **Issues :** GitHub Issues

---
*Dernière mise à jour : Août 2025*
*Version : 1.0.0*