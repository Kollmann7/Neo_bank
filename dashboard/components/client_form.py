import streamlit as st
from datetime import datetime

def render_client_form():
    """Formulaire de saisie client - Structure de votre main.py"""
    
    with st.form("client_scoring_form"):
        st.subheader("Informations client")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Finances")
            income = st.number_input("Revenus annuels (€)", 
                                   min_value=12000, value=50000, step=1000)
            credit_amount = st.number_input("Montant crédit (€)", 
                                          min_value=1000, value=200000, step=5000)
            annuity = st.number_input("Mensualités (€)", 
                                    min_value=100, value=1500, step=50)
        
        with col2:
            st.markdown("#### Personnel")
            age = st.number_input("Âge", min_value=18, max_value=80, value=35)
            employment_years = st.number_input("Années emploi", 
                                             min_value=0, max_value=50, value=5)
            gender = st.selectbox("Genre", ["M", "F"])
            family_size = st.number_input("Taille famille", 
                                        min_value=1, max_value=10, value=2)
        
        with col3:
            st.markdown("#### Complémentaires")
            children_ratio = st.slider("Ratio enfants", 0.0, 1.0, 0.5, 0.1)
            external_sources_mean = st.slider("Score externe", 0.0, 1.0, 0.6, 0.1)
            
            # Encodages (simplifiés pour les conseillers)
            education_level = st.selectbox("Éducation", 
                ["Primaire", "Secondaire", "Supérieur", "Master", "Doctorat"])
            income_type = st.selectbox("Type revenus", 
                ["Salarié", "Indépendant", "Pension", "Autre"])
            family_status = st.selectbox("Situation familiale", 
                ["Célibataire", "Marié", "Divorcé", "Veuf"])
        
        submitted = st.form_submit_button("Évaluer le Client", 
                                        use_container_width=True)
        
        if submitted:
            # Convertir en format API (comme votre main.py)
            client_data = {
                'client_id': f'client_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'income': income,
                'credit_amount': credit_amount,
                'annuity': annuity,
                'age': age,
                'employment_years': employment_years,
                'gender': gender,
                'family_size': family_size,
                'children_ratio': children_ratio,
                'external_sources_mean': external_sources_mean,
                # Conversion des sélections en encodages
                'education_encoded': ["Primaire", "Secondaire", "Supérieur", "Master", "Doctorat"].index(education_level),
                'income_type_encoded': ["Salarié", "Indépendant", "Pension", "Autre"].index(income_type),
                'family_status_encoded': ["Célibataire", "Marié", "Divorcé", "Veuf"].index(family_status)
            }
            
            return client_data
    
    return None