import streamlit as st
import numpy as np  # ✅ Ajout nécessaire
from components.api_client import api
from components.client_form import render_client_form
from components.risk_interpretation_simple import explain_score_for_advisor, create_score_gauge, explain_financial_ratios
from components.rgpd_compliance import show_rgpd_consent, show_privacy_notice, RGPDCompliantSession, validate_data_minimization

# Configuration
st.set_page_config(
    page_title="Neo-Bank Credit Dashboard",
    page_icon="🏦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2563eb 100%);
        padding: 1.5rem; 
        border-radius: 15px; 
        color: white; 
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8fafc;
    }
    
    /* Améliorer l'apparence des formulaires */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialisation de session RGPD
    RGPDCompliantSession.initialize_session()
    
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>Neo-Bank Credit Dashboard</h1>
        <p>Évaluation instantanée de crédit pour conseillers clientèle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérification du moteur de scoring intégré
    api_ok, api_info = api.health_check()
    
    if not api_ok:
        st.error("Moteur de scoring non disponible - Erreur de chargement du modèle")
        st.info("Solutions possibles :\n- Vérifiez que le fichier model/credit_model_v2.pkl existe\n- Redémarrez l'application\n- Contactez le support technique")
        
        # Affichage des détails d'erreur
        if 'error' in api_info:
            st.error(f"Détail de l'erreur : {api_info['error']}")
        
        st.stop()
    
    # Sidebar - État du moteur intégré
    st.sidebar.success("Moteur de scoring actif")
    st.sidebar.info(f"Version : {api_info.get('version', 'N/A')}")
    st.sidebar.info(f"Modèle : {'Chargé' if api_info.get('model_loaded') else 'Non chargé'}")
    st.sidebar.info(f"Features : {api_info.get('feature_count', 0)}")
    st.sidebar.info(f"RGPD : {'Actif' if api_info.get('rgpd_enabled') else 'Inactif'}")
    st.sidebar.success("Hébergé sur Streamlit Cloud")
    
    # Navigation simplifiée
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Choisir une action :", 
        ["Évaluer un Client", "Guide d'utilisation"],
        index=0
    )
    
    # PAGE PRINCIPALE : Scoring Client
    if page == "Évaluer un Client":
        st.header("Évaluation de Demande de Crédit")
        
        # Consentement RGPD obligatoire
        if not show_rgpd_consent():
            st.stop()
        
        # Instructions rapides
        with st.expander("💡 Instructions rapides", expanded=False):
            st.markdown("""
            **Comment utiliser ce dashboard :**
            1. 📝 Remplissez les informations du client dans le formulaire
            2. 🎯 Cliquez sur "Évaluer le Client"
            3. 📊 Consultez le score de risque et les recommandations
            4. 💼 Prenez votre décision en toute confiance
            """)
        
        st.markdown("---")
        
        # Formulaire principal - Utilise vos composants existants
        client_data = render_client_form()
        
        # Traitement du résultat
        if client_data:
            # Validation et minimisation RGPD
            minimized_data = validate_data_minimization(client_data)
            
            with st.spinner('🔄 **Évaluation en cours...** Analyse des 14 paramètres du modèle'):
                success, result = api.score_client(minimized_data)
            
            if success:
                # Animation de succès
                st.success("✅ **Évaluation terminée avec succès !**")
                
                st.markdown("---")
                st.markdown("## 📈 Résultats de l'Évaluation")
                
                # ✅ Gestion sécurisée de l'affichage
                try:
                    # Affichage intelligent pour conseiller
                    explain_score_for_advisor(result)
                    
                    st.markdown("---")
                    
                    # Graphiques côte à côte
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🎯 Score Visuel")
                        fig_gauge = create_score_gauge(result.get('risk_score', 0))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        # ✅ Vérification avant d'appeler explain_financial_ratios
                        if 'details' in result and result['details']:
                            explain_financial_ratios(result['details'])
                        else:
                            st.warning("⚠️ Détails financiers non disponibles dans la réponse API")
                            st.info("Les calculs de ratios seront affichés une fois que l'API retourne tous les détails")
                    
                    # Section détails techniques (repliable)
                    with st.expander("🔧 Détails techniques et données brutes", expanded=False):
                        col_tech1, col_tech2 = st.columns(2)
                        
                        with col_tech1:
                            st.markdown("**Données client (minimisées RGPD) :**")
                            st.json(minimized_data)
                        
                        with col_tech2:
                            st.markdown("**Réponse complète de l'API :**")
                            st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'affichage des résultats: {str(e)}")
                    st.info("💡 Les données sont disponibles mais il y a un problème d'affichage")
                    
                    # Affichage de base en cas d'erreur
                    st.json(result)

                # Actions suggérées
                st.markdown("---")
                st.markdown("### 🎯 Actions Suivantes")
                
                risk_score = result['risk_score']
                
                if risk_score < 35:
                    st.success("✅ **Dossier à approuver** - Risque faible, vous pouvez procéder en confiance.")
                    st.info("💡 **Conseil :** Préparez les documents de financement pour ce client.")
                elif risk_score < 65:
                    st.warning("⚠️ **Dossier à étudier** - Analyse complémentaire recommandée.")
                    st.info("💡 **Conseil :** Demandez des garanties supplémentaires ou justificatifs.")
                else:
                    st.error("❌ **Dossier à risque** - Refus recommandé ou conditions très strictes.")
                    st.info("💡 **Conseil :** Proposez des alternatives de financement ou orientez vers un autre produit.")
                
            else:
                st.error(f"❌ **Erreur lors de l'évaluation :** {result.get('error', 'Erreur inconnue')}")
                st.info("🔧 **Solutions :**\n- Vérifiez que l'API est démarrée\n- Contrôlez que tous les champs sont remplis\n- Contactez le support technique")
    
    # PAGE SECONDAIRE : Guide d'utilisation
    elif page == "Guide d'utilisation":
        st.header("Guide d'Utilisation du Dashboard")
        
        st.markdown("### Objectif")
        st.info("Ce dashboard permet d'évaluer en temps réel le risque de défaut d'un client pour une demande de crédit.")
        
        st.markdown("### Comment interpréter les scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🟢 Score 0-20% : TRÈS FAIBLE**
            - ✅ Approuver sans hésitation
            - Client excellent, risque minimal
            
            **🔵 Score 20-35% : FAIBLE**  
            - ✅ Approuver le dossier
            - Bon profil client
            
            **🟡 Score 35-50% : MODÉRÉ**
            - ⚠️ Approuver avec surveillance
            - Profil acceptable à surveiller
            """)
        
        with col2:
            st.markdown("""
            **🟠 Score 50-65% : ÉLEVÉ**
            - 🔍 Analyse approfondie requise
            - Profil à risque, étudier attentivement
            
            **🔴 Score 65-80% : TRÈS ÉLEVÉ**
            - ❌ Risque très important
            - Refus recommandé
            
            **⚫ Score 80%+ : CRITIQUE**
            - ❌ Refuser le dossier
            - Risque de défaut extrême
            """)
        
        st.markdown("### 💰 Ratios Financiers Importants")
        
        st.markdown("""
        **📊 Ratio Crédit/Revenus :**
        - 🟢 < 3x : Excellent (crédit raisonnable)
        - 🟡 3-5x : Correct (acceptable mais surveiller)  
        - 🔴 > 5x : Élevé (crédit très important vs revenus)
        
        **📈 Taux d'Endettement :**
        - 🟢 < 33% : Conforme aux standards bancaires
        - 🟡 33-50% : Limite haute acceptable
        - 🔴 > 50% : Dangereux pour le client
        """)
        
        st.markdown("### 🔒 Conformité RGPD")
        st.success("✅ **Ce dashboard respecte le RGPD :**")
        st.markdown("""
        - 🚫 **Aucun stockage** des données clients
        - ⚡ **Traitement en temps réel** uniquement
        - 🔐 **Moteur intégré sécurisé** sans API externe
        - 📝 **Logs anonymisés** pour le monitoring
        - 🎯 **Minimisation des données** automatique
        - 🛡️ **Consentement utilisateur** obligatoire
        """)
        
        # Notice de confidentialité
        show_privacy_notice()
        
        st.markdown("### 🆘 Support Technique")
        st.info("""
        **En cas de problème :**
        1. 🔄 Rechargez la page
        2. 🔌 Vérifiez que l'API est démarrée (`docker-compose up`)
        3. 📞 Contactez l'équipe technique avec le code d'erreur
        """)

if __name__ == "__main__":
    main()