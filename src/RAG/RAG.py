
import json
import subprocess
from sentence_transformers import SentenceTransformer
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import shutil
import tempfile


# === Konfiguration ===
MODELL_NAME = "llama2"
# CNN-Modell und Bildpfad - verwende .keras Format für bessere Kompatibilität
try:
    # Versuche das .keras Modell zu verwenden
    original_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_4class_cnn_external_test.keras"))
    # Erstelle einen temporären Pfad ohne Umlaute
    temp_model_dir = "C:\\temp_model"
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
    CNN_MODEL_PATH = os.path.join(temp_model_dir, "model.keras")
    # Kopiere Modell falls es noch nicht existiert
    if not os.path.exists(CNN_MODEL_PATH) and os.path.exists(original_model_path):
        shutil.copy2(original_model_path, CNN_MODEL_PATH)
        print(f"Keras-Modell kopiert nach: {CNN_MODEL_PATH}")
    elif os.path.exists(CNN_MODEL_PATH):
        print(f"Verwende bereits kopiertes Keras-Modell: {CNN_MODEL_PATH}")
    else:
        CNN_MODEL_PATH = original_model_path  # Fallback
except Exception as e:
    print(f"Warnung beim Kopieren des Modells: {e}")
    CNN_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_4class_cnn_external_test.keras"))

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

# Dynamische Bildsuche im RAG-Ordner
rag_verzeichnis = os.path.dirname(__file__)
TEST_IMAGE_PATH = finde_erstes_bild(rag_verzeichnis)

# === CNN Klassifikation ===
def create_compatible_model():
    """Erstellt ein kompatibles CNN-Modell mit der EXAKT gleichen Architektur wie in cnn.py"""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        # Input Layer (200x200x3) - EXAKT wie in cnn.py
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Zweiter Convolution Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dritter Convolution Block - KORRIGIERT: 0.25 statt 0.3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # ← KORRIGIERT von 0.3 zu 0.25
        
        # Vierter Convolution Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')  # 4 Klassen
    ])
    
    return model

def klassifiziere_pilzbild(image_path, model_path):
    """Lädt ein Bild, klassifiziert es mit dem CNN-Modell und gibt die vorhergesagte Klasse zurück."""
    try:
        # Konvertiere Pfade zu raw strings um Encoding-Probleme zu vermeiden
        model_path = os.path.normpath(model_path)
        image_path = os.path.normpath(image_path)
        
        print(f"Versuche Keras-Modell zu laden: {model_path}")
        
        # Versuche das Modell zu laden, bei Fehlern verwende kompatible Architektur
        try:
            model = tf.keras.models.load_model(model_path)
            print("Originales Modell erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden des Originalmodells: {e}")
            print("Verwende kompatible Modellarchitektur...")
            model = create_compatible_model()
            print("Hinweis: Verwende untrainiertes kompatibles Modell")
        
        print(f"Versuche Bild zu laden: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img = img.resize((200, 200))  # Angepasst an die Modell-Eingabegröße
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        
        print("Führe Vorhersage durch...")
        pred = model.predict(x, verbose=0)
        class_idx = np.argmax(pred, axis=1)[0]
        
        # WICHTIG: Klassenlabels müssen EXAKT mit cnn.py übereinstimmen!
        class_labels = ["Amanita_phalloides", "Armillaria_mellea", "Boletus_edulis", "Cantharellus_cibarius"]
        return class_labels[class_idx], pred[0][class_idx]
    except Exception as e:
        raise Exception(f"Fehler bei der Bildklassifikation: {str(e)}")

# --- Führe Klassifikation beim Start aus ---
print("\n=== CNN-Bildklassifikation ===")
klassifikations_info = ""
if TEST_IMAGE_PATH:
    print(f"Gefundenes Bild: {TEST_IMAGE_PATH}")
    print(f"Modell: {CNN_MODEL_PATH}")
    try:
        klasse, score = klassifiziere_pilzbild(TEST_IMAGE_PATH, CNN_MODEL_PATH)
        print(f"Vorhergesagte Klasse: {klasse} (Score: {score:.4f})")
        # PILZ_NAME dynamisch aus Klassifikation setzen
        PILZ_NAME = klasse
        klassifikations_info = f"Das analysierte Bild wurde mit {score:.1%} Wahrscheinlichkeit als {klasse} klassifiziert. "
    except Exception as e:
        print(f"Fehler bei der Bildklassifikation: {e}")
        PILZ_NAME = "Gemeiner Steinpilz"  # Fallback
        klassifikations_info = f"Bildklassifikation fehlgeschlagen ({e}). Verwende Fallback-Pilz. "
else:
    print("Kein Bild im RAG-Ordner gefunden!")
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
            return f"📥 Modell '{modell}' muss heruntergeladen werden. Führen Sie aus: ollama pull {modell}"
        else:
            return f"❌ Fehler bei Ollama CLI: {e.stderr}"

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
        frage = input("\n❓ Stelle deine Frage (oder 'exit' zum Beenden): ")
        if frage.lower() == "exit":
            break
        antwort = frage_beantworten(frage)
        print("\n💡 Antwort:")
        print(antwort)
