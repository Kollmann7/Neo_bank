import subprocess
import sys
import os

def check_dependencies():
    """Vérifier et installer les dépendances nécessaires"""
    required_packages = {
        'streamlit': 'streamlit>=1.28.0',
        'requests': 'requests>=2.31.0', 
        'plotly': 'plotly>=5.15.0',
        'numpy': 'numpy>=1.24.0',
        'pandas': 'pandas>=2.0.0'
    }
    
    missing_packages = []
    
    for package, version in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} déjà installé")
        except ImportError:
            missing_packages.append(version)
            print(f"❌ {package} manquant")
    
    if missing_packages:
        print(f"\n📦 Installation de {len(missing_packages)} packages manquants...")
        for package in missing_packages:
            print(f"Installation de {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Toutes les dépendances sont installées!")
    else:
        print("✅ Toutes les dépendances sont déjà présentes!")

def main():
    print("🏦 Neo-Bank Dashboard - Démarrage Automatique")
    print("="*50)
    
    # Vérifier le répertoire
    if not os.path.exists("app.py"):
        print("❌ Erreur: app.py non trouvé")
        print("💡 Lancez ce script depuis le dossier dashboard/")
        input("Appuyez sur Entrée pour quitter...")
        return
    
    try:
        # Vérifier et installer les dépendances
        check_dependencies()
        
        print("\n🚀 Lancement du dashboard...")
        print("🌐 URL: http://localhost:8501")
        print("⚠️  Assurez-vous que l'API est démarrée sur http://localhost:8000")
        print("="*50)
        
        # Lancer Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        input("Appuyez sur Entrée pour quitter...")

if __name__ == "__main__":
    main()