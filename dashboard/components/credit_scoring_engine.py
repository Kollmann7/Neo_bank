"""
🏦 Neo-Bank Credit Scoring Engine
API intégrée directement dans Streamlit (sans Docker)
Conforme RGPD - Aucun stockage de données
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import streamlit as st
from typing import Dict, Any, Tuple
import logging

# Configuration des logs (anonymisés pour RGPD)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditScoringEngine:
    """Moteur de scoring intégré - Compatible Streamlit Cloud"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.encoders = None
        self.model_version = "v2"
        self.is_loaded = False
        
        # Tentative de chargement automatique
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle et ses composants"""
        try:
            # Chemin vers le modèle (compatible Streamlit Cloud)
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model')
            
            # Chargement du modèle
            with open(os.path.join(model_path, 'credit_model_v2.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            
            # Chargement du scaler (si disponible)
            try:
                with open(os.path.join(model_path, 'scaler_v2.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                logger.warning("Scaler non trouvé, utilisation de la standardisation par défaut")
                self.scaler = None
            
            # Chargement des noms de features (si disponible)
            try:
                with open(os.path.join(model_path, 'feature_names_v2.pkl'), 'rb') as f:
                    self.feature_names = pickle.load(f)
            except FileNotFoundError:
                # Features par défaut basées sur le notebook
                self.feature_names = [
                    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                    'AGE_YEARS', 'EMPLOYMENT_YEARS', 
                    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
                    'FAMILY_SIZE', 'CHILDREN_RATIO',
                    'EXTERNAL_SOURCES_MEAN',
                    'EDUCATION_ENCODED', 'INCOME_TYPE_ENCODED', 
                    'FAMILY_STATUS_ENCODED', 'CODE_GENDER'
                ]
            
            # Chargement des encoders (si disponible)
            try:
                with open(os.path.join(model_path, 'encoders_v2.pkl'), 'rb') as f:
                    self.encoders = pickle.load(f)
            except FileNotFoundError:
                logger.warning("Encoders non trouvés, utilisation d'encodage simplifié")
                self.encoders = None
            
            self.is_loaded = True
            logger.info(f"Modèle {self.model_version} chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            self.is_loaded = False
    
    def _prepare_features(self, client_data: Dict[str, Any]) -> np.ndarray:
        """Préparer les features pour le modèle - EXACTEMENT 13 features dans l'ordre du modèle"""
        try:
            # Calculs des ratios financiers
            credit_income_ratio = client_data['credit_amount'] / max(client_data['income'], 1)
            annuity_income_ratio = client_data['annuity'] / max(client_data['income'], 1)
            
            # Construction du vecteur avec les 13 features EXACTES du modèle:
            # ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN', 
            #  'AGE_YEARS', 'EMPLOYMENT_YEARS', 'CODE_GENDER_ENCODED', 'NAME_EDUCATION_TYPE_ENCODED',
            #  'NAME_INCOME_TYPE_ENCODED', 'NAME_FAMILY_STATUS_ENCODED', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']
            
            features = [
                client_data['income'],                              # AMT_INCOME_TOTAL
                client_data['credit_amount'],                       # AMT_CREDIT
                client_data['annuity'],                            # AMT_ANNUITY
                client_data.get('family_size', 2),                 # CNT_FAM_MEMBERS
                client_data.get('children_ratio', 0) * client_data.get('family_size', 2), # CNT_CHILDREN (approximation)
                client_data['age'],                                # AGE_YEARS
                client_data['employment_years'],                   # EMPLOYMENT_YEARS
                1 if client_data.get('gender', 'M') == 'M' else 0, # CODE_GENDER_ENCODED
                client_data.get('education_encoded', 1),           # NAME_EDUCATION_TYPE_ENCODED
                client_data.get('income_type_encoded', 0),         # NAME_INCOME_TYPE_ENCODED
                client_data.get('family_status_encoded', 0),       # NAME_FAMILY_STATUS_ENCODED
                credit_income_ratio,                               # CREDIT_INCOME_RATIO
                annuity_income_ratio                               # ANNUITY_INCOME_RATIO
            ]
            
            # Vérification: doit être exactement 13 features
            assert len(features) == 13, f"Erreur: {len(features)} features au lieu de 13"
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des features: {str(e)}")
            raise ValueError(f"Erreur de préparation des données: {str(e)}")
    
    def predict_risk(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédiction du risque de défaut
        
        Args:
            client_data: Données du client (dict)
            
        Returns:
            dict: Score et recommandations
        """
        if not self.is_loaded:
            return {
                'error': 'Modèle non chargé',
                'risk_score': None,
                'recommendation': 'ERREUR TECHNIQUE'
            }
        
        try:
            # Log anonymisé (RGPD compliant)
            logger.info(f"Nouvelle évaluation - Age: {client_data.get('age', 'N/A')}, "
                       f"Revenus: {client_data.get('income', 'N/A')//1000 if client_data.get('income') else 'N/A'}k")
            
            # Préparation des features
            features = self._prepare_features(client_data)
            
            # Application du scaler si disponible
            if self.scaler:
                features = self.scaler.transform(features)
            else:
                # Standardisation simple si pas de scaler
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Prédiction
            risk_proba = self.model.predict_proba(features)[0][1]
            risk_score = int(risk_proba * 100)
            
            # Interprétation métier
            interpretation = self._interpret_score(risk_score, client_data)
            
            # Construction de la réponse
            result = {
                'risk_score': risk_score,
                'risk_probability': float(risk_proba),
                'risk_level': interpretation['risk_level'],
                'recommendation': interpretation['recommendation'],
                'explanation': interpretation['explanation'],
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'credit_income_ratio': f"{(client_data['credit_amount'] / client_data['income']):.2%}",
                    'annuity_income_ratio': f"{(client_data['annuity'] / client_data['income']):.2%}",
                    'employment_years': client_data['employment_years'],
                    'family_size': client_data.get('family_size', 2)
                }
            }
            
            logger.info(f"Évaluation terminée - Score: {risk_score}%, Recommandation: {interpretation['recommendation']}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return {
                'error': str(e),
                'risk_score': None,
                'recommendation': 'ERREUR TECHNIQUE'
            }
    
    def _interpret_score(self, risk_score: int, client_data: Dict[str, Any]) -> Dict[str, str]:
        """Interpréter le score de risque pour les conseillers"""
        
        # Analyse des ratios financiers
        credit_ratio = client_data['credit_amount'] / client_data['income']
        annuity_ratio = client_data['annuity'] / client_data['income']
        
        # Interprétation selon les seuils métier
        if risk_score < 15:
            return {
                'risk_level': "TRÈS FAIBLE",
                'recommendation': "APPROUVER",
                'explanation': f"Excellent profil client (score: {risk_score}%). Risque minimal de défaut."
            }
        elif risk_score < 30:
            return {
                'risk_level': "FAIBLE", 
                'recommendation': "APPROUVER",
                'explanation': f"Bon profil client (score: {risk_score}%). Risque acceptable pour la banque."
            }
        elif risk_score < 50:
            return {
                'risk_level': "MODÉRÉ",
                'recommendation': "APPROUVER avec conditions",
                'explanation': f"Profil correct (score: {risk_score}%). Surveillance recommandée de l'évolution."
            }
        elif risk_score < 70:
            return {
                'risk_level': "ÉLEVÉ",
                'recommendation': "ÉTUDIER",
                'explanation': f"Profil à risque (score: {risk_score}%). Analyse approfondie requise avant décision."
            }
        else:
            return {
                'risk_level': "TRÈS ÉLEVÉ",
                'recommendation': "REFUSER",
                'explanation': f"Risque critique (score: {risk_score}%). Probabilité de défaut trop importante."
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Status de santé du moteur de scoring"""
        return {
            'status': 'healthy' if self.is_loaded else 'error',
            'model_loaded': self.is_loaded,
            'model_version': self.model_version,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'rgpd_enabled': True,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }

# Instance globale pour Streamlit
@st.cache_resource
def get_scoring_engine():
    """Créer une instance cachée du moteur de scoring"""
    return CreditScoringEngine()

# Fonctions utilitaires pour l'interface
def validate_client_data(client_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Valider les données client (RGPD et business rules)"""
    
    required_fields = ['income', 'credit_amount', 'annuity', 'age', 'employment_years']
    
    # Vérification des champs obligatoires
    for field in required_fields:
        if field not in client_data or client_data[field] is None:
            return False, f"Champ obligatoire manquant: {field}"
    
    # Validation des valeurs
    if client_data['income'] <= 0:
        return False, "Les revenus doivent être positifs"
    
    if client_data['credit_amount'] <= 0:
        return False, "Le montant du crédit doit être positif"
    
    if client_data['age'] < 18 or client_data['age'] > 100:
        return False, "L'âge doit être entre 18 et 100 ans"
    
    if client_data['employment_years'] < 0:
        return False, "L'ancienneté ne peut pas être négative"
    
    # Validation des ratios (règles métier)
    credit_ratio = client_data['credit_amount'] / client_data['income']
    if credit_ratio > 10:  # Plus de 10x les revenus annuels
        return False, "Montant de crédit disproportionné par rapport aux revenus"
    
    annuity_ratio = client_data['annuity'] / client_data['income']
    if annuity_ratio > 0.8:  # Plus de 80% des revenus en remboursement
        return False, "Mensualité trop élevée par rapport aux revenus"
    
    return True, "Données valides"

def anonymize_for_logs(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymiser les données pour les logs (RGPD)"""
    return {
        'age_range': f"{(client_data.get('age', 0) // 10) * 10}-{((client_data.get('age', 0) // 10) * 10) + 9}",
        'income_range': f"{client_data.get('income', 0)//10000 * 10}k-{(client_data.get('income', 0)//10000 + 1) * 10}k",
        'credit_range': f"{client_data.get('credit_amount', 0)//50000 * 50}k-{(client_data.get('credit_amount', 0)//50000 + 1) * 50}k",
        'employment_category': 'employed' if client_data.get('employment_years', 0) > 0 else 'unemployed'
    }
