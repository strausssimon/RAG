"""
====================================================
Programmname : RAG mit CNN Version 0.1
Beschreibung : RAG (Retrieval-Augmented Generation) zur Klassifikation von Pilzen und Informationsbereitstellung

====================================================
"""
import json
import numpy as np
import faiss
import subprocess
import shutil
import os
import re
import cv2
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# === Pfade ===
PILZ_DATEI = os.path.join(os.path.dirname(__file__), "Informationen_RAG.json")
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mushroom_4class_cnn_external_test.h5")
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "armillaria_mellea_19915.jpg")

def clean_output(text):
    """Entfernt ANSI-Escape-Codes und andere Formatierungen aus der Ausgabe"""
    if not text:
        return ""
    
    # ANSI-Escape-Codes entfernen
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', text)
    
    # Zusätzliche Bereinigungen
    cleaned = cleaned.replace('\r', '')  # Carriage Returns entfernen
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)  # Steuerzeichen entfernen
    
    return cleaned.strip()

def check_ollama_installed():
    """Prüft ob Ollama installiert ist"""
    # Verschiedene mögliche Pfade für Ollama prüfen
    possible_paths = [
        "ollama",  # Standard PATH
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.environ.get('USERNAME', '')),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    
    for path in possible_paths:
        if shutil.which(path) or (path.endswith('.exe') and os.path.exists(path)):
            print(f"Ollama gefunden: {path}")
            return path
    
    print("Ollama ist nicht installiert oder nicht im PATH!")
    print("Installieren Sie Ollama von: https://ollama.ai/download")
    return None

# === Embedding-Modell laden ===
print("Lade Embedding-Modell...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === CNN-Modell laden ===
print("Lade CNN-Modell für Pilzklassifikation...")

def create_compatible_model():
    """Erstellt ein kompatibles CNN-Modell mit der gleichen Architektur"""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        # Input Layer (200x200x3)
        layers.Input(shape=(200, 200, 3)),
        
        # Erste Convolution Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Zweiter Convolution Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dritter Convolution Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
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

cnn_model = None
if os.path.exists(CNN_MODEL_PATH):
    print(f"Modell gefunden: {CNN_MODEL_PATH}")
    print(f" Dateigröße: {os.path.getsize(CNN_MODEL_PATH) / (1024*1024):.1f} MB")
    
    try:
        print("Versuche Standard-Lademethode...")
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
        print(f"CNN-Modell erfolgreich geladen!")
    except Exception as e1:
        print(f"Standard-Methode fehlgeschlagen: Kompatibilitätsproblem")
        print(" Verwende Gewichte-Transfer-Methode...")
        
        try:
            # Erstelle kompatibles Modell und lade nur die Gewichte
            print("Erstelle kompatible Modellarchitektur...")
            cnn_model = create_compatible_model()
            
            # Lade nur die Gewichte aus der H5-Datei
            print("Lade Gewichte aus H5-Datei...")
            cnn_model.load_weights(CNN_MODEL_PATH)
            print(f"CNN-Modell mit Gewichte-Transfer geladen!")
            
        except Exception as e2:
            print(f"Gewichte-Transfer fehlgeschlagen: {e2}")
            print("Das Modell ist nicht kompatibel mit dieser TensorFlow-Version")
            cnn_model = None
else:
    print(f"CNN-Modell nicht gefunden: {CNN_MODEL_PATH}")
    print("Stellen Sie sicher, dass das Training abgeschlossen ist!")

# Debug-Information über das geladene Modell
if cnn_model is not None:
    print(f"Modell-Typ: {type(cnn_model)}")
    print(f"Modell-Input-Shape: {cnn_model.input_shape}")
    print(f"Modell-Output-Shape: {cnn_model.output_shape}")
    print(f"Anzahl Layer: {len(cnn_model.layers)}")

# Definiere die Klassennamen wie im ursprünglichen Modell
class_names = ["Amanita_phalloides", "Armillaria_mellea", "Boletus_edulis", "Cantharellus_cibarius"]

# === Pilzdaten laden ===
print("Lade Pilzdaten...")
with open(PILZ_DATEI, "r", encoding="utf-8") as f:
    pilzdaten = json.load(f)

texte = []
for pilz in pilzdaten:
    text = f"""Pilz: {pilz['name']}
Essbar: {pilz.get('essbar', 'unbekannt')}
Beschreibung: {pilz.get('beschreibung', 'keine Beschreibung vorhanden')}
Zubereitung: {pilz.get('zubereitung', 'keine Angabe')}
"""
    texte.append(text)

# === Embeddings berechnen & FAISS-Index aufbauen ===
print("Erstelle FAISS-Index...")
embeddings = embedder.encode(texte, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Ollama CLI Abfrage ===
def frage_mit_ollama(prompt, modell="phi3:mini"):
    # Erst prüfen ob Ollama verfügbar ist
    ollama_path = check_ollama_installed()
    if not ollama_path:
        return "Ollama nicht verfügbar"
    
    try:
        result = subprocess.run(
            [ollama_path, "run", modell, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",     # UTF-8 Encoding setzen
            errors="replace",     # Ersetze unbekannte Zeichen statt Fehler
            check=True,
        )
        return clean_output(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        if "pull" in str(e.stderr):
            return f"Modell '{modell}' muss heruntergeladen werden. Führen Sie aus: ollama pull {modell}"
        else:
            return f"Fehler bei Ollama CLI: {e.stderr}"

# === Bildklassifikation ===
def klassifiziere_pilzbild(bild_pfad):
    """Klassifiziert ein Pilzbild mit dem CNN-Modell"""
    if cnn_model is None:
        return None, 0.0, "CNN-Modell nicht verfügbar"
    
    if not os.path.exists(bild_pfad):
        return None, 0.0, f"Bild nicht gefunden: {bild_pfad}"
    
    try:
        # Bild laden und vorverarbeiten (wie im ursprünglichen CNN)
        img = cv2.imread(bild_pfad)
        if img is None:
            return None, 0.0, f"Kann Bild nicht laden: {bild_pfad}"
        
        # Auf 200x200 resizen (wie im Training)
        img = cv2.resize(img, (200, 200))
        img = img / 255.0  # Normalisierung
        
        # Batch-Dimension hinzufügen
        img_batch = np.expand_dims(img, axis=0)
        
        # Vorhersage
        prediction = cnn_model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class = class_names[predicted_class_idx]
        
        print(f"Bildklassifikation:")
        print(f"Erkannter Pilz: {predicted_class}")
        print(f"Konfidenz: {confidence:.2%}")
        
        return predicted_class, confidence, "Erfolg"
    
    except Exception as e:
        return None, 0.0, f"Fehler bei Bildklassifikation: {str(e)}"

# === RAG Fragebeantwortung ===
def frage_beantworten(frage, top_k=3):
    frage_embedding = embedder.encode([frage], convert_to_numpy=True)
    _, indices = index.search(frage_embedding, top_k)
    kontext = "\n\n".join([texte[i] for i in indices[0]])

    prompt = (
    f"Du bist ein deutschsprachiger Pilzexperte. Nutze ausschließlich den folgenden Kontext, um die Frage zu beantworten. "
    f"Wenn die Antwort nicht im Kontext enthalten ist, sage 'Ich habe dazu keine Information.'\n\n"
    f"=== Kontext ===\n{kontext}\n\n"
    f"=== Frage ===\n{frage}\n\n"
    f"=== Antwort (auf Deutsch): ==="
    )

    antwort = frage_mit_ollama(prompt)
    return antwort

# === Bildbasierte RAG Fragebeantwortung ===
def bild_frage_beantworten(bild_pfad, zusatz_frage=""):
    """Beantwortet Fragen zu einem Pilzbild"""
    # 1. Bild klassifizieren
    predicted_class, confidence, status = klassifiziere_pilzbild(bild_pfad)
    
    if predicted_class is None:
        return f"Bildklassifikation fehlgeschlagen: {status}"
    
    # 2. Relevante Pilzinformationen für RAG suchen
    pilz_name = predicted_class.replace("_", " ")  # z.B. "Amanita_phalloides" -> "Amanita phalloides"
    
    # Suche spezifisch nach dem erkannten Pilz
    such_query = f"{pilz_name} Pilz Beschreibung Essbarkeit"
    if zusatz_frage:
        such_query += f" {zusatz_frage}"
    
    frage_embedding = embedder.encode([such_query], convert_to_numpy=True)
    _, indices = index.search(frage_embedding, 3)  # Top 3 relevante Einträge
    kontext = "\n\n".join([texte[i] for i in indices[0]])
    
    # 3. Prompt für Ollama erstellen
    if zusatz_frage:
        haupt_frage = f"Was ist das für ein Pilz und {zusatz_frage}"
    else:
        haupt_frage = "Was ist das für ein Pilz? Beschreibe ihn detailliert."
    
    prompt = (
        f"Du bist ein deutschsprachiger Pilzexperte. Ein Bild wurde automatisch als '{pilz_name}' "
        f"klassifiziert (Konfidenz: {confidence:.1%}). "
        f"Nutze den folgenden Kontext und die Klassifikation, um die Frage zu beantworten.\n\n"
        f"=== Bildklassifikation ===\n"
        f"Erkannter Pilz: {pilz_name}\n"
        f"Konfidenz: {confidence:.1%}\n\n"
        f"=== Kontext aus Pilzdatenbank ===\n{kontext}\n\n"
        f"=== Frage ===\n{haupt_frage}\n\n"
        f"=== Antwort (auf Deutsch): ==="
    )
    
    antwort = frage_mit_ollama(prompt)
    return antwort

# === Main ===
if __name__ == "__main__":
    # Test mit dem bereitgestellten Bild
    test_bild = "armillaria_mellea_19915.jpg"
    test_bild_pfad = os.path.join(os.path.dirname(__file__), test_bild)
    
    if os.path.exists(test_bild_pfad):
        print(f"Analysiere Testbild: {test_bild}")
        print("=" * 60)
        
        # Bildbasierte Analyse
        antwort = bild_frage_beantworten(test_bild_pfad)
        print("\nKI-Analyse des Pilzbildes:")
        print("-" * 60)
        print(antwort)
        print("-" * 60)
        
        # Optional: Zusätzliche Frage
        print("\n" + "=" * 60)
        print("Möchten Sie eine zusätzliche Frage zu diesem Pilz stellen?")
        zusatz_frage = input("Zusatzfrage (oder Enter zum Überspringen): ").strip()
        
        if zusatz_frage:
            antwort_zusatz = bild_frage_beantworten(test_bild_pfad, zusatz_frage)
            print("\nAntwort auf Zusatzfrage:")
            print("-" * 60)
            print(antwort_zusatz)
            print("-" * 60)
    else:
        print(f"Testbild nicht gefunden: {test_bild_pfad}")
        print("Stelle stattdessen eine normale Textfrage:")
        
        print("Stelle deine Frage zu Pilzen:")
        frage = input("> ")
        antwort = frage_beantworten(frage)
        print("\nAntwort von Phi3:mini via Ollama:")
        print("-" * 60)
        print(antwort)
        print("-" * 60)