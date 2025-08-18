
# TensorFlow 2.19.1 & Keras 3.11.2 Konfiguration - EXKLUSIV
import os
# Moderne TensorFlow/Keras 3 Konfiguration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # OneDNN Optimierungen falls problematisch

# Standard Imports
import json
import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import glob
import shutil

# EXKLUSIV Keras 3.11.2 - Keine Fallbacks
try:
    import keras
    import tensorflow as tf
    
    # Überprüfe Keras-Version - nur 3.x erlaubt
    keras_version = keras.__version__
    major_version = int(keras_version.split('.')[0])
    
    if major_version < 3:
        raise ImportError(f"Keras {keras_version} ist nicht unterstützt. Mindestens Keras 3.x erforderlich!")
    
    KERAS_AVAILABLE = True
    print(f"✅ Keras 3.x erfolgreich geladen (Version: {keras.__version__})")
    print(f"✅ TensorFlow erfolgreich geladen (Version: {tf.__version__})")
    
except ImportError as e:
    print(f"❌ KRITISCHER FEHLER: Keras 3.x ist erforderlich!")
    print(f"❌ Fehlerdetails: {e}")
    print(f"❌ Installieren Sie Keras 3.x mit: pip install keras>=3.11.2")
    print("🛑 Script wird beendet - Keras 3.x ist zwingend erforderlich!")
    exit(1)
except Exception as e:
    print(f"❌ UNERWARTETER FEHLER beim Keras 3.x Import: {e}")
    print("🛑 Script wird beendet!")
    exit(1)


# === Konfiguration ===
MODELL_NAME = "llama2"
# CNN-Modell Pfade - versuche mehrere Formate
CNN_MODEL_KERAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_4class_cnn_external_test.keras"))
CNN_MODEL_H5 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_4class_cnn_external_test.h5"))
PILZ_DATEI = os.path.join(os.path.dirname(__file__), "Informationen_RAG.json")

def check_ollama_installed():
    """Prüft ob Ollama installiert ist - aus RAG.py übernommen"""
    possible_paths = [
        "ollama",  # Standard PATH
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.environ.get('USERNAME', '')),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    
    for path in possible_paths:
        if shutil.which(path) or (path.endswith('.exe') and os.path.exists(path)):
            return path
    return "ollama"  # Fallback

# === Dynamische Bildsuche ===
def finde_erstes_bild(verzeichnis):
    """Sucht nach der ersten JPG- oder PNG-Datei im angegebenen Verzeichnis."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    for ext in extensions:
        files = glob.glob(os.path.join(verzeichnis, ext))
        if files:
            return files[0]
    return None

# CNN Modell laden - EXKLUSIV für Keras 3.x
def load_cnn_model(model_paths=None):
    """
    Lädt das CNN-Modell mit Keras 3.x.
    Unterstützt .keras und .h5 Formate.
    
    Args:
        model_paths (list): Liste von Modellpfaden oder None für Standard-Pfade
        
    Returns:
        model: Geladenes Keras-Modell oder None bei Fehler
    """
    # Standard-Pfade wenn keine angegeben
    if model_paths is None:
        model_paths = [CNN_MODEL_KERAS, CNN_MODEL_H5]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"⚠️ Modell nicht gefunden: {os.path.basename(model_path)}")
            continue
            
        print(f"Versuche Modell zu laden: {os.path.basename(model_path)}")
        
        # Keras 3.x Standard-Methode
        try:
            model = keras.models.load_model(model_path)
            print(f"✅ Modell erfolgreich geladen mit Keras 3.x: {os.path.basename(model_path)}")
            return model
        except Exception as e1:
            print(f"Keras 3.x Standard-Laden fehlgeschlagen: {str(e1)[:100]}...")
            
            # Keras 3.x ohne Kompilierung (für Kompatibilität)
            try:
                model = keras.models.load_model(model_path, compile=False)
                print(f"✅ Modell erfolgreich geladen (compile=False) mit Keras 3.x: {os.path.basename(model_path)}")
                return model
            except Exception as e2:
                print(f"Keras 3.x compile=False fehlgeschlagen: {str(e2)[:100]}...")
                continue
    
    print("❌ Alle verfügbaren Modell-Formate fehlgeschlagen mit Keras 3.x")
    return None

# Dynamische Bildsuche im RAG-Ordner
rag_verzeichnis = os.path.dirname(__file__)
TEST_IMAGE_PATH = finde_erstes_bild(rag_verzeichnis)

# === CNN Klassifikation ===
def klassifiziere_pilzbild(image_path, model_paths=None):
    """
    Klassifiziert ein Pilzbild mit dem CNN-Modell.
    EXKLUSIV für Keras 3.x.
    """
    try:
        # Verwende die Keras 3.x Lade-Methode
        model = load_cnn_model(model_paths)
        
        # Prüfe ob das Modell erfolgreich geladen wurde
        if model is None:
            raise Exception("Modell konnte nicht geladen werden - alle verfügbaren Formate versucht")
        
        print(f"Lade und verarbeite Bild: {os.path.basename(image_path)}")
        
        # Bildverarbeitung für Keras 3.x
        img = Image.open(image_path).convert('RGB')
        img = img.resize((200, 200))
        
        # Konvertierung zu NumPy Array mit korrekter Normalisierung
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalisierung auf [0,1]
        
        print("Führe CNN-Vorhersage durch...")
        
        # Vorhersage mit Keras 3.x
        pred = model.predict(x, verbose=0)
            
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][class_idx])
        
        # Aktualisierte Klassenlabels
        class_labels = ["Amanita_phalloides", "Armillaria_mellea", "Boletus_edulis", "Cantharellus_cibarius"]
        predicted_class = class_labels[class_idx]
        
        print(f"✅ Vorhersage: {predicted_class} (Konfidenz: {confidence:.4f})")
        return predicted_class, confidence
        
    except Exception as e:
        raise Exception(f"Fehler bei der Bildklassifikation: {str(e)}")

# --- Führe Klassifikation beim Start aus ---
print("\n=== CNN-Bildklassifikation (TensorFlow 2.19.1 & Keras 3.x EXKLUSIV) ===")
print("🚀 Keras 3.x erfolgreich initialisiert")

klassifikations_info = ""
if TEST_IMAGE_PATH:
    print(f"Gefundenes Bild: {os.path.basename(TEST_IMAGE_PATH)}")
    print(f"Verfügbare Modellformate: .keras (bevorzugt), .h5 (Fallback)")
    try:
        klasse, score = klassifiziere_pilzbild(TEST_IMAGE_PATH)
        print(f"✅ Vorhergesagte Klasse: {klasse} (Konfidenz: {score:.1%})")
        # PILZ_NAME dynamisch aus Klassifikation setzen
        PILZ_NAME = klasse
        klassifikations_info = f"Das analysierte Bild wurde mit {score:.1%} Wahrscheinlichkeit als {klasse} klassifiziert. "
    except Exception as e:
        print(f"❌ Fehler bei der Bildklassifikation: {e}")
        PILZ_NAME = "Gemeiner Steinpilz"  # Fallback
        klassifikations_info = f"Bildklassifikation fehlgeschlagen ({e}). Verwende Fallback-Pilz. "
else:
    print("ℹ️ Kein Bild im RAG-Ordner gefunden!")
    PILZ_NAME = "Gemeiner Steinpilz"  # Fallback
    klassifikations_info = "Kein Bild gefunden. Verwende Fallback-Pilz. "
    PILZ_NAME = "Gemeiner Steinpilz"  # Fallback
    klassifikations_info = "Kein Bild zur Analyse gefunden. Verwende Fallback-Pilz. "

# === Embedding-Modell laden ===
print("Lade Embedding-Modell...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Pilzdaten laden ===
print("Lade Pilzdaten...")
with open(PILZ_DATEI, "r", encoding="utf-8") as f:
    pilzdaten = json.load(f)

# === Gewählten Pilz finden ===
pilz_info = next((p for p in pilzdaten if p["bezeichnung"]["name"] == PILZ_NAME), None)
if pilz_info is None:
    raise ValueError(f"Pilz '{PILZ_NAME}' nicht in der JSON-Datei gefunden!")

# === Vorstellung direkt aus JSON-Daten generieren ===
hut = pilz_info["aussehen"].get("hut", "unbekannt")
stiel = pilz_info["aussehen"].get("stiel", "unbekannt")
lamellen = pilz_info["aussehen"].get("schwamm_lamellen", "unbekannt")
essbar = pilz_info["verzehr"].get("essbar", "unbekannt")

vorstellung_text = (
    f"Der Pilz ist der {PILZ_NAME}. "
    f"Sein Hut ist {hut}, der Stiel ist {stiel} und er hat {lamellen}. "
    f"Hinsichtlich der Essbarkeit gilt: {essbar}."
)

print("\n=== Vorstellung des gewählten Pilzes ===")
print(vorstellung_text)

# === Kontext nur mit gewähltem Pilz ===
kontext = json.dumps(pilz_info, ensure_ascii=False, indent=2)

# === Ollama CLI Abfrage ===
def frage_mit_ollama(prompt, modell=MODELL_NAME):
    ollama_path = check_ollama_installed()
    try:
        result = subprocess.run(
            [ollama_path, "run", modell, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Geändert von "ignore" zu "replace" wie in RAG.py
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if "pull" in str(e.stderr):
            return f"Modell '{modell}' muss heruntergeladen werden. Führen Sie aus: ollama pull {modell}"
        else:
            return f"Fehler bei Ollama CLI: {e.stderr}"

# === Fragebeantwortung nur für gewählten Pilz ===
def frage_beantworten(frage):
    prompt = (
        "Du bist ein deutschsprachiger Pilzexperte. "
        "Nutze ausschließlich den folgenden Kontext, um die Frage zu beantworten. "
        "Antwort immer in vollständigen, klar formulierten Sätzen. "
        "Füge keine Informationen hinzu, die nicht im Kontext enthalten sind. "
        "Wenn die Antwort nicht im Kontext steht, sage: "
        "'Leider habe ich dazu keine Information.'\n\n"
        f"=== Bildanalyse ===\n{klassifikations_info}\n\n"
        f"Es geht um den Pilz: {PILZ_NAME}\n\n"
        f"=== Kontext ===\n{kontext}\n\n"
        f"=== Frage ===\n{frage}\n\n"
        "=== Antwort (auf Deutsch, vollständige Sätze): ==="
    )
    return frage_mit_ollama(prompt)

# === Interaktive Schleife ===
if __name__ == "__main__":
    while True:
        frage = input("\nStellen Sie Ihre Frage (oder 'exit' zum Beenden): ")
        if frage.lower() == "exit":
            break
        antwort = frage_beantworten(frage)
        print("\nAntwort:")
        print(antwort)
