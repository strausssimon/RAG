import json
import subprocess
from sentence_transformers import SentenceTransformer
import os
import shutil

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

# === Konfiguration ===
#PILZ_NAME = "Armillaria mellea"  # <-- Hier Pilznamen ändern (passend zum Testbild)
MODELL_NAME = "phi3:mini"  # Gleiches Modell wie in RAG.py
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mushroom_4class_cnn_external_test.h5")

# === Pfade ===
PILZ_DATEI = os.path.join(os.path.dirname(__file__), "Informationen_RAG.json")

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