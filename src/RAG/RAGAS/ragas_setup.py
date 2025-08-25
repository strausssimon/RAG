"""
====================================================
Programmname : RAGAS Setup
Datum        : 17.08.2025
Version      : 1.0
Beschreibung : Installation und Setup für RAGAS Evaluation
 
====================================================
"""
import subprocess
import sys
import os

def install_ragas():
    """Installiert RAGAS und Abhängigkeiten"""
    print("📦 Installiere RAGAS und Abhängigkeiten...")
    
    packages = [
        "ragas",
        "datasets",
        "openai",  # Für OpenAI API (optional)
        "langchain",  # Für lokale Modelle (optional)
        "huggingface_hub"
    ]
    
    for package in packages:
        try:
            print(f"   Installiere {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package} installiert")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Fehler bei {package}: {e}")

def setup_environment():
    """Setup für Umgebungsvariablen"""
    print("\n🔧 Environment Setup...")
    
    # Prüfe OpenAI API Key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️  OPENAI_API_KEY nicht gesetzt!")
        print("   Für beste RAGAS-Performance setze:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("   oder verwende lokale Modelle")
    else:
        print("✅ OPENAI_API_KEY gefunden")

def test_imports():
    """Testet ob alle Imports funktionieren"""
    print("\n🧪 Teste Imports...")
    
    try:
        import ragas
        print("✅ RAGAS importiert")
        
        from ragas.metrics import faithfulness, answer_relevancy
        print("✅ RAGAS Metriken verfügbar")
        
        from datasets import Dataset
        print("✅ Datasets verfügbar")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False

def main():
    """Hauptfunktion für Setup"""
    print("🛠️  RAGAS SETUP FÜR RAG EVALUATION")
    print("=" * 50)
    
    # Installation
    install_ragas()
    
    # Environment Setup
    setup_environment()
    
    # Test Imports
    success = test_imports()
    
    if success:
        print("\n🎉 RAGAS Setup erfolgreich!")
        print("Du kannst jetzt ragas_evaluation.py ausführen")
    else:
        print("\n😞 Setup nicht vollständig")
        print("Überprüfe die Fehlermeldungen oben")

if __name__ == "__main__":
    main()
