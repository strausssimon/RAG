import json
import numpy as np
import faiss
import subprocess
from sentence_transformers import SentenceTransformer

# === Pfade ===
PILZ_DATEI = r"C:\Users\Angle\OneDrive\Desktop\Masterstudium\3. Semester\Big Data Analyseprojekt\SLM mit RAG\Informationen_RAG.json"

# === Embedding-Modell laden ===
print("🔢 Lade Embedding-Modell...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Pilzdaten laden ===
print("📂 Lade Pilzdaten...")
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
print("📈 Erstelle FAISS-Index...")
embeddings = embedder.encode(texte, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

OLLAMA_PATH = r"C:\Users\Angle\AppData\Local\Programs\Ollama\ollama.exe"

# === Ollama CLI Abfrage ===
def frage_mit_ollama(prompt, modell="llama2"):
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", modell, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",     # UTF-8 Encoding setzen
            errors="ignore",      # Fehler ignorieren (optional)
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei Ollama CLI: {e.stderr}")
        return ""

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

# === Main ===
if __name__ == "__main__":
    print("❓ Stelle deine Frage zu Pilzen:")
    frage = input("> ")
    antwort = frage_beantworten(frage)
    print("\n💡 Antwort von Llama2 via Ollama:")
    print(antwort)