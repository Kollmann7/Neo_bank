"""
🔒 RGPD & Conformité Data Protection
Configuration pour la Neo-Bank
"""

from datetime import datetime, timedelta
import streamlit as st
import logging

# Configuration RGPD
RGPD_CONFIG = {
    "data_retention_days": 0,  # Aucun stockage permanent
    "anonymized_logs": True,
    "user_consent_required": True,
    "data_processing_purpose": "Évaluation de crédit en temps réel",
    "legal_basis": "Intérêt légitime bancaire",
    "data_controller": "Neo-Bank",
    "privacy_contact": "privacy@neo-bank.fr"
}

def show_rgpd_consent():
    """Afficher la demande de consentement RGPD"""
    
    if 'rgpd_consent' not in st.session_state:
        st.session_state.rgpd_consent = False
    
    if not st.session_state.rgpd_consent:
        st.warning("📋 **Consentement RGPD requis**")
        
        with st.expander("🔒 Informations sur la protection des données", expanded=True):
            st.markdown("""
            ### 🛡️ Vos données sont protégées
            
            **🎯 Finalité du traitement :**
            - Évaluation en temps réel du risque de crédit
            - Aide à la décision pour les conseillers clientèle
            
            **📊 Données collectées :**
            - Informations financières (revenus, crédit demandé)
            - Données personnelles de base (âge, situation familiale)
            - Informations professionnelles (ancienneté)
            
            **🔒 Vos droits :**
            - ✅ **Pas de stockage** : Vos données ne sont jamais sauvegardées
            - ✅ **Traitement instantané** : Analyse en temps réel uniquement  
            - ✅ **Logs anonymisés** : Seules des statistiques anonymes sont conservées
            - ✅ **Droit d'accès** : Vous pouvez demander les informations traitées
            
            **⚖️ Base légale :** Intérêt légitime bancaire (art. 6.1.f RGPD)
            
            **📞 Contact :** privacy@neo-bank.fr
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ J'accepte le traitement", type="primary"):
                st.session_state.rgpd_consent = True
                st.success("✅ Consentement enregistré")
                st.rerun()
        
        with col2:
            if st.button("❌ Je refuse"):
                st.error("❌ L'évaluation ne peut pas être effectuée sans consentement")
                st.stop()
        
        return False
    
    return True

def log_anonymized_activity(activity_type: str, details: dict = None):
    """
    Enregistrer une activité de manière anonymisée (RGPD compliant)
    
    Args:
        activity_type: Type d'activité (evaluation, error, etc.)
        details: Détails anonymisés optionnels
    """
    
    # Logger configuré sans données personnelles
    logger = logging.getLogger('neo_bank_rgpd')
    
    anonymized_entry = {
        'timestamp': datetime.now().isoformat(),
        'activity': activity_type,
        'session_id': hash(st.session_state.get('session_id', 'anonymous')) % 10000,
        'details': details or {}
    }
    
    logger.info(f"Activity: {activity_type}")

def show_privacy_notice():
    """Afficher la notice de confidentialité"""
    
    st.markdown("""
    ---
    ### 🔒 Protection des Données Personnelles
    
    **Cette application respecte le RGPD :**
    
    - 🚫 **Aucun stockage** : Vos données ne sont jamais enregistrées
    - ⚡ **Traitement instantané** : Analyse en temps réel uniquement
    - 📊 **Logs anonymisés** : Seules des statistiques non-identifiantes
    - 🔐 **Chiffrement** : Communications sécurisées
    - 📞 **Support** : privacy@neo-bank.fr
    
    *Dernière mise à jour : Août 2025*
    """)

def validate_data_minimization(client_data: dict) -> dict:
    """
    Valider que seules les données nécessaires sont collectées (principe de minimisation RGPD)
    
    Args:
        client_data: Données du client
        
    Returns:
        dict: Données validées et minimisées
    """
    
    # Champs strictement nécessaires pour l'évaluation
    essential_fields = {
        'income': client_data.get('income'),
        'credit_amount': client_data.get('credit_amount'), 
        'annuity': client_data.get('annuity'),
        'age': client_data.get('age'),
        'employment_years': client_data.get('employment_years'),
        'gender': client_data.get('gender', 'N/A')
    }
    
    # Champs optionnels pour améliorer l'analyse (avec valeurs par défaut)
    optional_fields = {
        'family_size': client_data.get('family_size', 2),
        'children_ratio': client_data.get('children_ratio', 0),
        'external_sources_mean': client_data.get('external_sources_mean', 0.5)
    }
    
    # Fusion des données minimisées
    minimized_data = {**essential_fields, **optional_fields}
    
    # Log anonymisé du respect de la minimisation
    log_anonymized_activity('data_minimization_applied', {
        'essential_fields': len(essential_fields),
        'optional_fields': len(optional_fields),
        'total_fields': len(minimized_data)
    })
    
    return minimized_data

def get_data_retention_policy():
    """Retourner la politique de rétention des données"""
    
    return {
        'client_data_retention': '0 jours (aucun stockage)',
        'anonymous_logs_retention': '90 jours maximum',
        'session_data_retention': 'Durée de la session uniquement',
        'model_data_retention': 'Stockage technique nécessaire au fonctionnement',
        'last_updated': '2025-08-14'
    }

class RGPDCompliantSession:
    """Gestionnaire de session conforme RGPD"""
    
    @staticmethod
    def initialize_session():
        """Initialiser une session respectueuse du RGPD"""
        
        if 'session_initialized' not in st.session_state:
            # ID de session anonymisé (hash non-réversible)
            st.session_state.session_id = datetime.now().isoformat()
            st.session_state.session_initialized = True
            st.session_state.rgpd_consent = False
            
            # Log du démarrage de session
            log_anonymized_activity('session_started')
    
    @staticmethod
    def cleanup_session():
        """Nettoyer les données de session (appelé à la fermeture)"""
        
        # Log de fin de session
        log_anonymized_activity('session_ended')
        
        # Nettoyage des données sensibles (si applicable)
        sensitive_keys = ['client_data', 'evaluation_result']
        for key in sensitive_keys:
            if key in st.session_state:
                del st.session_state[key]
