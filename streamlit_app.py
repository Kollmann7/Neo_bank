"""
🏦 Neo-Bank Credit Dashboard
Point d'entrée principal pour Streamlit Cloud
Déployé depuis GitHub
"""

import sys
import os

# Ajouter le répertoire dashboard au path pour les imports
dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard')
sys.path.append(dashboard_path)

# Import et lancement de l'app principale
from dashboard.app import main

if __name__ == "__main__":
    main()
