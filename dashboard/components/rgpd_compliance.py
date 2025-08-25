"""
🔒 RGPD & Conformité Data Protection
Configuration pour la Neo-Bank
"""

from datetime import datetime
import logging
import uuid
from typing import Optional, Dict

import streamlit as st


# Configuration RGPD
RGPD_CONFIG = {
    "data_retention_days": 0,  # Aucun stockage permanent des données clients (traitement en temps réel)
    "anonymized_logs_retention_days": 90,
    "user_consent_required": True,
    "data_processing_purpose": "Évaluation de crédit en temps réel",
    "legal_basis": "Intérêt légitime bancaire",
    "data_controller": "Neo-Bank",
    "privacy_contact": "privacy@neo-bank.fr",
    "privacy_policy_url": "https://www.cnil.fr/fr/reglement-europeen-protection-donnees",
}


# Logger basique pour activités anonymisées
logger = logging.getLogger("neo_bank_rgpd")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _ensure_session_keys():
    if "rgpd" not in st.session_state:
        st.session_state.rgpd = {}
    rgpd = st.session_state.rgpd
    if "session_anonym_id" not in rgpd:
        # identifiant anonymisé et non réversible pour les logs (UUID)
        rgpd["session_anonym_id"] = uuid.uuid4().hex[:12]
    if "created_at" not in rgpd:
        rgpd["created_at"] = datetime.utcnow().isoformat()


def record_consent(granted: bool):
    """Enregistre le consentement dans la session (éphemère)"""
    _ensure_session_keys()
    st.session_state.rgpd["consent"] = bool(granted)
    st.session_state.rgpd["consent_at"] = datetime.utcnow().isoformat()
    st.session_state.rgpd["consent_version"] = "v1.0"
    log_anonymized_activity("consent_recorded", {"granted": granted})


def has_consent() -> bool:
    return bool(st.session_state.get("rgpd", {}).get("consent", False))


def show_rgpd_consent() -> bool:
    """Afficher la demande de consentement RGPD et stocker le choix en session."""
    _ensure_session_keys()

    if not RGPD_CONFIG["user_consent_required"]:
        return True

    if has_consent():
        return True

    st.warning("📋 **Consentement RGPD requis**")
    with st.expander("🔒 Informations sur la protection des données", expanded=True):
        st.markdown(
            f"""
- Finalité : **{RGPD_CONFIG['data_processing_purpose']}**
- Base légale : **{RGPD_CONFIG['legal_basis']}**
- Contrôleur : **{RGPD_CONFIG['data_controller']}**
- Contact confidentialité : **{RGPD_CONFIG['privacy_contact']}**
- Pour plus d'information : [Règlement (RGPD) - CNIL]({RGPD_CONFIG['privacy_policy_url']})
"""
        )
        st.markdown(
            "- Vos droits : accès, rectification, effacement, limitation, opposition et portabilité.\n"
            f"- Conservation : {get_data_retention_policy()['client_data_retention']}."
        )

    cols = st.columns([3, 1])
    with cols[0]:
        agree = st.checkbox(
            "J'autorise le traitement de mes données pour l'évaluation de crédit", key="rgpd_checkbox"
        )
    with cols[1]:
        if st.button("Enregistrer le consentement"):
            if agree:
                record_consent(True)
                st.success("Consentement enregistré — vous pouvez continuer.")
                return True
            else:
                st.error("Vous devez cocher la case pour donner votre consentement.")
                record_consent(False)
                return False
    return False


def log_anonymized_activity(activity_type: str, details: Optional[dict] = None):
    """
    Enregistrer une activité de manière anonymisée (RGPD compliant)

    Args:
        activity_type: Type d'activité (evaluation, error, etc.)
        details: Détails anonymisés optionnels
    """
    _ensure_session_keys()
    anon_id = st.session_state.rgpd.get("session_anonym_id", "anon")
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "activity": activity_type,
        "anon_id": anon_id,
        "details": details or {},
    }
    # Enregistrer uniquement des métadonnées non identifiantes
    logger.info(f"RGPD-ACTIVITY {entry}")


def show_privacy_notice():
    """Afficher la notice de confidentialité"""
    st.markdown(
        """
---
### 🔒 Protection des données personnelles

**Cette application respecte les principes du RGPD :**

- **Aucun stockage permanent** des données clients (traitement en temps réel) sauf fichiers techniques nécessaires au modèle.
- **Minimisation** : seules les données strictement nécessaires sont utilisées.
- **Logs anonymisés** : utilisation d'identifiants non réversibles.
- **Chiffrement** : communications via TLS (déploiement) et bonnes pratiques d'accès.
- **Droits** : Vous pouvez demander l'accès, la rectification, l'effacement ou la limitation des données traitées.
- **Contact** : privacy@neo-bank.fr

Pour plus d'informations : https://www.cnil.fr/fr/reglement-europeen-protection-donnees

*Dernière mise à jour : Août 2025*
"""
    )


def validate_data_minimization(client_data: dict) -> dict:
    """
    Valider que seules les données nécessaires sont collectées (principe de minimisation RGPD)
    N'accepte le traitement que si le consentement utilisateur est présent (si requis).

    Args:
        client_data: Données du client

    Returns:
        dict: Données validées et minimisées
    """
    # Si le consentement est requis et absent, lever une exception ou renvoyer {}
    if RGPD_CONFIG["user_consent_required"] and not has_consent():
        log_anonymized_activity("attempt_process_without_consent", {})
        raise PermissionError("Consentement RGPD requis pour traiter ces données.")

    # Champs strictement nécessaires pour l'évaluation
    essential_fields = {
        "income": client_data.get("income"),
        "credit_amount": client_data.get("credit_amount"),
        "annuity": client_data.get("annuity"),
        "age": client_data.get("age"),
        "employment_years": client_data.get("employment_years"),
        "gender": client_data.get("gender", "N/A"),
    }

    # Champs optionnels pour améliorer l'analyse (avec valeurs par défaut)
    optional_fields = {
        "family_size": client_data.get("family_size", 2),
        "children_ratio": client_data.get("children_ratio", 0),
        "external_sources_mean": client_data.get("external_sources_mean", 0.5),
    }

    # Fusion des données minimisées
    minimized_data = {**essential_fields, **optional_fields}

    # Log anonymisé du respect de la minimisation
    log_anonymized_activity(
        "data_minimization_applied",
        {
            "essential_fields": len(essential_fields),
            "optional_fields": len(optional_fields),
            "total_fields": len(minimized_data),
        },
    )

    return minimized_data


def get_data_retention_policy() -> Dict[str, str]:
    """Retourner la politique de rétention des données"""
    return {
        "client_data_retention": "0 jours (aucun stockage des données clients après la session)",
        "anonymous_logs_retention": f"{RGPD_CONFIG['anonymized_logs_retention_days']} jours maximum",
        "session_data_retention": "Durée de la session uniquement",
        "model_data_retention": "Stockage technique nécessaire au fonctionnement (pas de données personnelles)",
        "last_updated": "2025-08-14",
    }


class RGPDCompliantSession:
    """Gestionnaire de session conforme RGPD"""

    @staticmethod
    def initialize_session():
        """Initialise les clés de session nécessaires et un identifiant anonymisé."""
        _ensure_session_keys()
        # Marquer l'heure d'initialisation
        st.session_state.rgpd["initialized_at"] = datetime.utcnow().isoformat()
        log_anonymized_activity("session_initialized", {})

    @staticmethod
    def cleanup_session():
        """Nettoyer toutes les données sensibles en fin de session."""
        if "rgpd" in st.session_state:
            # Conserver seulement les infos non sensibles ou supprimer tout si policy l'exige
            preserved = {"session_anonym_id": st.session_state.rgpd.get("session_anonym_id")}
            # supprimer la clé rgpd et réécrire minimalement
            st.session_state.rgpd.clear()
            st.session_state.rgpd.update(preserved)
        # Supprimer autres clés sensibles si présentes
        for k in list(st.session_state.keys()):
            if k.startswith("sensitive_") or k in {"client_raw_data", "client_personal_id"}:
                del st.session_state[k]
        log_anonymized_activity("session_cleaned", {})

