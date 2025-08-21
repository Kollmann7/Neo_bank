"""
🏦 Neo-Bank API Client - Version intégrée sans Docker
Compatible Streamlit Cloud avec GitHub
"""

import streamlit as st
from typing import Dict, Any, Tuple
from .credit_scoring_engine import get_scoring_engine, validate_client_data

class NeoBank_API:
    """API Client intégrée """
    
    def __init__(self):
        # Chargement du moteur de scoring intégré
        self.scoring_engine = get_scoring_engine()
    
    def health_check(self) -> Tuple[bool, Dict]:
        """Vérifier l'état du moteur de scoring intégré"""
        try:
            health_status = self.scoring_engine.get_health_status()
            return health_status['status'] == 'healthy', health_status
        except Exception as e:
            return False, {'error': str(e), 'status': 'error'}
    
    def score_client(self, client_data: Dict[str, Any]) -> Tuple[bool, Dict]:
        """
        Scorer un client via le moteur intégré
        
        Args:
            client_data: Données du client
            
        Returns:
            Tuple[bool, Dict]: (succès, résultat ou erreur)
        """
        try:
            # Validation des données (RGPD et business)
            is_valid, error_msg = validate_client_data(client_data)
            if not is_valid:
                return False, {'error': f'Données invalides: {error_msg}'}
            
            # Scoring via moteur intégré
            result = self.scoring_engine.predict_risk(client_data)
            
            # Vérification du résultat
            if 'error' in result:
                return False, result
            
            return True, result
            
        except Exception as e:
            return False, {'error': f'Erreur technique: {str(e)}'}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Informations sur le modèle chargé"""
        return self.scoring_engine.get_health_status()

# Instance globale
api = NeoBank_API()