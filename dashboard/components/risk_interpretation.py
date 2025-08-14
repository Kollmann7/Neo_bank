import streamlit as st
import plotly.graph_objects as go

def explain_score_for_advisor(result: dict):
    """Explication enrichie avec transparence sur les ajustements"""
    
    risk_score = result.get('risk_score', 0)
    base_score = result.get('base_score', risk_score)
    risk_level = result.get('risk_level', 'INCONNU')
    
    # Configuration couleurs
    risk_config = {
        "TRÈS FAIBLE": {"color": "#10b981", "icon": "🟢", "action": "✅ Approuver sans hésitation"},
        "FAIBLE": {"color": "#3b82f6", "icon": "🔵", "action": "✅ Approuver le dossier"},
        "MODÉRÉ": {"color": "#f59e0b", "icon": "🟡", "action": "⚠️ Approuver avec surveillance"},
        "ÉLEVÉ": {"color": "#ef4444", "icon": "🟠", "action": "🔍 Analyse approfondie requise"},
        "TRÈS ÉLEVÉ": {"color": "#dc2626", "icon": "🔴", "action": "❌ Risque très important"},
        "CRITIQUE": {"color": "#7f1d1d", "icon": "⚫", "action": "❌ Refuser le dossier"}
    }
    
    config = risk_config.get(risk_level, risk_config["MODÉRÉ"])
    
    # Affichage principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # ✅ Affichage du score avec transparence
        if base_score != risk_score:
            st.markdown(f"""
            <div style="background: {config['color']}; color: white; padding: 2rem; 
                        border-radius: 15px; text-align: center;">
                <h1>{config['icon']}</h1>
                <h2>{risk_score}%</h2>
                <h3>{risk_level}</h3>
                <p style="font-size: 0.8em; opacity: 0.9;">
                    Score ML: {base_score}% → Ajusté: {risk_score}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: {config['color']}; color: white; padding: 2rem; 
                        border-radius: 15px; text-align: center;">
                <h1>{config['icon']}</h1>
                <h2>{risk_score}%</h2>
                <h3>{risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Ce que cela signifie :")
        st.info(result.get('explanation', 'Aucune explication disponible'))
        
        st.markdown("### 🎯 Action recommandée :")
        st.success(config['action'])
        
        # ✅ Explication des ajustements
        if 'details' in result and 'score_adjustments' in result['details']:
            adjustments = result['details']['score_adjustments']
            if adjustments:
                st.markdown("### 📊 Ajustements appliqués :")
                for adj in adjustments:
                    st.info(f"• {adj}")
        
        # Alertes spécifiques
        if 'alerts' in result and result['alerts']:
            st.markdown("### ⚠️ Points d'attention :")
            for alert in result['alerts']:
                if alert.startswith('📊'):
                    st.info(alert)  # Ajustements en bleu
                else:
                    st.warning(alert)  # Alertes en orange

def create_score_gauge(risk_score: int):
    """Jauge de score visuelle"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Risque"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 35], 'color': "green"},
                {'range': [35, 50], 'color': "yellow"},
                {'range': [50, 65], 'color': "orange"},
                {'range': [65, 80], 'color': "red"},
                {'range': [80, 100], 'color': "darkred"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def explain_financial_ratios(details: dict):
    """Analyse financière avec cohérence score/ratios"""
    
    st.markdown("### 📊 Analyse Financière Détaillée")
    
    if not details or not isinstance(details, dict):
        st.warning("⚠️ Détails financiers non disponibles")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratio crédit/revenus avec impact sur score
        credit_ratio_str = details.get('credit_income_ratio', '0x')
        
        try:
            credit_ratio = float(credit_ratio_str.replace('x', '').replace(',', '.').strip())
            
            # ✅ Évaluation cohérente avec l'ajustement de score
            if credit_ratio <= 3:
                ratio_status = "🟢 Excellent"
                ratio_explain = "Ratio idéal, aucun impact négatif sur le score"
                impact = "Aucun ajustement"
            elif credit_ratio <= 5:
                ratio_status = "🟡 Surveillé"
                ratio_explain = "Ratio acceptable mais surveiller l'évolution"
                impact = "Score ajusté +5 points"
            else:
                ratio_status = "🔴 Préoccupant"
                ratio_explain = "Ratio élevé, impact significatif sur le risque"
                impact = f"Score ajusté +{min(15, int(credit_ratio * 2))} points"
            
            st.metric("Crédit/Revenus", f"{credit_ratio:.1f}x")
            st.markdown(f"**{ratio_status}**")
            st.markdown(f"*{ratio_explain}*")
            st.caption(f"Impact: {impact}")
            
        except (ValueError, AttributeError):
            st.metric("Crédit/Revenus", "N/A")
            st.warning("⚠️ Impossible de calculer le ratio")
    
    with col2:
        # Taux d'endettement avec impact sur score
        debt_ratio_str = details.get('annuity_income_ratio', '0%')
        
        try:
            debt_ratio = float(debt_ratio_str.replace('%', '').replace(',', '.').strip())
            
            # ✅ Évaluation cohérente avec l'ajustement de score
            if debt_ratio <= 33:
                debt_status = "🟢 Conforme"
                debt_explain = "Respecte les recommandations bancaires (≤33%)"
                impact = "Aucun ajustement"
            elif debt_ratio <= 50:
                debt_status = "🟡 Limite"
                debt_explain = "Proche de la limite recommandée, surveiller"
                impact = "Score ajusté +10 points"
            else:
                debt_status = "🔴 Critique"
                debt_explain = "Dépasse largement les recommandations (>50%)"
                impact = "Score ajusté +20 points"
            
            st.metric("Taux d'endettement", f"{debt_ratio:.1f}%")
            st.markdown(f"**{debt_status}**")
            st.markdown(f"*{debt_explain}*")
            st.caption(f"Impact: {impact}")
            
        except (ValueError, AttributeError):
            st.metric("Taux d'endettement", "N/A")
            st.warning("⚠️ Impossible de calculer le ratio")
    
    # ✅ Section explicative
    st.markdown("---")
    st.markdown("### 🧠 Comment le Score est Calculé")
    
    st.info("""
    **Le score final combine :**
    1. 🤖 **Score du modèle ML** (14 variables analysées)
    2. 📊 **Ajustements financiers** (ratios critiques)
    3. ⚖️ **Pondération intelligente** pour cohérence métier
    
    Cette approche garantit que les ratios financiers préoccupants 
    se reflètent correctement dans le score final.
    """)