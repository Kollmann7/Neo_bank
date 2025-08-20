import streamlit as st
import plotly.graph_objects as go

def explain_score_for_advisor(result: dict):
    """Explication simplifiée pour Figma"""
    
    risk_score = result.get('risk_score', 0)
    risk_level = result.get('risk_level', 'INCONNU')
    
    # Configuration couleurs (sans emojis)
    risk_config = {
        "TRÈS FAIBLE": {"color": "#10b981", "action": "Approuver sans hésitation"},
        "FAIBLE": {"color": "#3b82f6", "action": "Approuver le dossier"},
        "MODÉRÉ": {"color": "#f59e0b", "action": "Approuver avec surveillance"},
        "ÉLEVÉ": {"color": "#ef4444", "action": "Analyse approfondie requise"},
        "TRÈS ÉLEVÉ": {"color": "#dc2626", "action": "Risque très important"},
        "CRITIQUE": {"color": "#7f1d1d", "action": "Refuser le dossier"}
    }
    
    config = risk_config.get(risk_level, risk_config["MODÉRÉ"])
    
    # Affichage principal simplifié
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Affichage du score épuré
        st.markdown(f"""
        <div style="background: {config['color']}; color: white; padding: 2rem; 
                    border-radius: 15px; text-align: center;">
            <h2>{risk_score}%</h2>
            <h3>{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Ce que cela signifie :")
        st.info(result.get('explanation', 'Aucune explication disponible'))
        
        st.markdown("### Action recommandée :")
        st.success(config['action'])
        
        # Détails financiers simplifiés
        if 'details' in result:
            details = result['details']
            st.markdown("### Détails financiers :")
            st.write(f"**Ratio crédit/revenus :** {details.get('credit_income_ratio', 'N/A')}")
            st.write(f"**Ratio mensualité/revenus :** {details.get('annuity_income_ratio', 'N/A')}")
            st.write(f"**Années d'emploi :** {details.get('employment_years', 'N/A')}")

def create_score_gauge(score: int, risk_level: str = None) -> go.Figure:
    """Gauge épurée pour Figma"""
    
    # Couleurs simplifiées
    if score < 30:
        color = "#10b981"  # Vert
    elif score < 50:
        color = "#3b82f6"  # Bleu
    elif score < 70:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Rouge
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Risque", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#f0fdf4"},
                {'range': [30, 50], 'color': "#eff6ff"},
                {'range': [50, 70], 'color': "#fffbeb"},
                {'range': [70, 100], 'color': "#fef2f2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=0, r=0))
    return fig

def explain_financial_ratios(details: dict):
    """Explication simplifiée des ratios basée sur les détails de l'API"""
    
    st.markdown("### Analyse des Ratios Financiers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Affichage du ratio crédit/revenus (déjà calculé)
        credit_ratio_str = details.get('credit_income_ratio', 'N/A')
        st.metric(
            label="Ratio Crédit/Revenus", 
            value=credit_ratio_str,
            help="Montant du crédit par rapport aux revenus annuels"
        )
        
        # Extraction de la valeur numérique pour l'interprétation
        try:
            if credit_ratio_str != 'N/A':
                # Conversion de "400.00%" en nombre
                credit_ratio = float(credit_ratio_str.replace('%', '')) / 100
                
                # Interprétation
                if credit_ratio <= 3:
                    st.success("Ratio excellent")
                elif credit_ratio <= 5:
                    st.info("Ratio acceptable")
                elif credit_ratio <= 7:
                    st.warning("Ratio élevé")
                else:
                    st.error("Ratio très élevé")
        except (ValueError, AttributeError):
            st.info("Calcul automatique non disponible")
    
    with col2:
        # Affichage du taux d'endettement (déjà calculé)
        annuity_ratio_str = details.get('annuity_income_ratio', 'N/A')
        st.metric(
            label="Taux d'Endettement", 
            value=annuity_ratio_str,
            help="Pourcentage des revenus consacré au remboursement"
        )
        
        # Extraction de la valeur numérique pour l'interprétation
        try:
            if annuity_ratio_str != 'N/A':
                # Conversion de "30.00%" en nombre
                annuity_ratio = float(annuity_ratio_str.replace('%', '')) / 100
                
                # Interprétation
                if annuity_ratio <= 0.33:
                    st.success("Taux acceptable")
                elif annuity_ratio <= 0.45:
                    st.warning("Taux élevé")
                else:
                    st.error("Taux critique")
        except (ValueError, AttributeError):
            st.info("Calcul automatique non disponible")
    
    # Informations additionnelles si disponibles
    if 'employment_years' in details:
        st.markdown(f"**Ancienneté professionnelle :** {details['employment_years']} ans")
    if 'family_size' in details:
        st.markdown(f"**Taille de la famille :** {details['family_size']} personnes")