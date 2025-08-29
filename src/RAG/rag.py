"""
====================================================
Programmname : RAG mit CNN
Beschreibung : RAG (Retrieval-Augmented Generation) zur Klassifikation von Pilzen und Informationsbereitstellung

====================================================
"""
import json
import subprocess
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import glob
import shutil
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import cv2

# === Konfiguration ===
MODELL_NAME = "llama2"
original_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "mushroom_5class_resnet_cnn_80_20_split_2.keras"))
modell = tf.keras.models.load_model(original_model_path)
PILZ_DATEI = os.path.join(os.path.dirname(__file__), "Informationen_RAG.json")

# === Globale Variablen ===
PILZ_NAME = None
klassifikations_info = ""
kontext = ""
faiss_index = None
embed_model = None
pilzdaten = None
pilz_texts = None

pilz_embeddings = None

# === Testmodus für RAG-Tests ===
def setze_test_pilz(pilzname: str):
    """Setzt PILZ_NAME und kontext für Testzwecke, ohne Bildklassifikation."""
    global PILZ_NAME, kontext
    with open(PILZ_DATEI, "r", encoding="utf-8") as f:
        pilzdaten = json.load(f)
    pilz_info = next((p for p in pilzdaten if p["bezeichnung"]["name"].lower() == pilzname.lower()), None)
    if pilz_info is not None:
        PILZ_NAME = pilzname
        kontext = json.dumps(pilz_info, ensure_ascii=False, indent=2)
    else:
        PILZ_NAME = None
        kontext = ""

# === Hilfsfunktionen ===
def check_ollama_installed():
    possible_paths = [
        "ollama",
        rf"C:\\Users\\{os.environ.get('USERNAME', '')}\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
        r"C:\\Program Files\\Ollama\\ollama.exe",
        r"C:\\Program Files (x86)\\Ollama\\ollama.exe"
    ]
    for path in possible_paths:
        if shutil.which(path) or (path.endswith('.exe') and os.path.exists(path)):
            return path
    return "ollama"

def frage_mit_ollama(prompt, modell=MODELL_NAME):
    ollama_path = check_ollama_installed()
    try:
        result = subprocess.run(
            [ollama_path, "run", modell, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        text = result.stdout.strip()
        if text:
            return text
        
    except subprocess.CalledProcessError as e:
        if "pull" in str(e.stderr):
            return f"Modell '{modell}' muss heruntergeladen werden. Bitte: ollama pull {modell}"
        else:
            return f"Fehler bei Ollama CLI: {e.stderr}"

# Lade FAISS und Embeddings asynchron
def lade_embeddings_async():
    global faiss_index, embed_model, pilzdaten, pilz_texts, pilz_embeddings
    from sentence_transformers import SentenceTransformer
    import faiss

    with open(PILZ_DATEI, "r", encoding="utf-8") as f:
        pilzdaten = json.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    pilz_texts = [json.dumps(p, ensure_ascii=False) for p in pilzdaten]
    pilz_embeddings = np.array([embed_model.encode(t, convert_to_numpy=True, normalize_embeddings=True) for t in pilz_texts]).astype("float32")

    embedding_dim = pilz_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(pilz_embeddings)

# === Bild Klassifikation ===
def bereite_bild_vor(image_path):
    with open(image_path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Kann Bild nicht laden: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (200, 200))
    img_normalized = img_resized.astype(np.float32) / 255.0
    x = np.expand_dims(img_normalized, axis=0)
    return x

def klassifiziere_pilzbild(image_path, modell):
    eingabe = bereite_bild_vor(image_path)
    vorhersage = modell.predict(eingabe)
    klasse_index = np.argmax(vorhersage)
    wahrscheinlichkeit = vorhersage[0][klasse_index]
    pilzklassen = ["Stinkmorchel", "Fliegenpilz", "Gemeiner Steinpilz", "Echter Pfifferling", "Hallimasch"]
    erkannter_pilz = pilzklassen[klasse_index]
    return erkannter_pilz, wahrscheinlichkeit

# === RAG Initialisierung ===
def initialisiere_rag_mit_bild(image_path=None):
    global PILZ_NAME, klassifikations_info, kontext

    if not image_path or not os.path.exists(image_path):
        PILZ_NAME = None
        klassifikations_info = "Kein Bild zur Analyse gefunden."
        return {"klassifikation": klassifikations_info, "pilzname": PILZ_NAME, "beschreibung": "Keine Analyse möglich."}

    try:
        klasse, score = klassifiziere_pilzbild(image_path, modell)
        PILZ_NAME = klasse
        if score < 0.6:
            PILZ_NAME = None
            klassifikations_info = "Der Pilz wurde nicht erkannt. Bitte ein neues Bild hochladen."
            return {"klassifikation": klassifikations_info, "pilzname": PILZ_NAME, "beschreibung": ""}
        else:
            klassifikations_info = f"Der Pilz wurde mit {score:.1%} Wahrscheinlichkeit als {klasse} erkannt."
    except Exception as e:
        PILZ_NAME = None
        klassifikations_info = f"Bildklassifikation fehlgeschlagen: {e}."
        return {"klassifikation": klassifikations_info, "pilzname": PILZ_NAME, "beschreibung": ""}

    with open(PILZ_DATEI, "r", encoding="utf-8") as f:
        pilzdaten = json.load(f)

    pilz_info = next((p for p in pilzdaten if p["bezeichnung"]["name"] == PILZ_NAME), None)

    if pilz_info is None:
        return {"klassifikation": klassifikations_info, "pilzname": PILZ_NAME, "beschreibung": "Dazu liegen mir keine Informationen vor."}

    kontext = json.dumps(pilz_info, ensure_ascii=False, indent=2)

    hut = pilz_info["aussehen"].get("hut", "unbekannt")
    stiel = pilz_info["aussehen"].get("stiel", "unbekannt")
    lamellen = pilz_info["aussehen"].get("schwamm_lamellen", "unbekannt")
    essbar = pilz_info["verzehr"].get("essbar", "unbekannt")

    beschreibung_prompt = (
        "Du bist ein deutschsprachiger Pilzexperte und antwortest mit Sätzen wie aus einem Sachbuch."
        "Schreibe einen Fließtext und keine Stichpunkte. Die Sätze dürfen die Informationen ausschließlich aus folgendem Kontext beinhalten und keine weiteren Informationen."
        f"=== Kontext ===\n"
        f"Bescheibe den Hut vom {PILZ_NAME}: {hut}\n"
        f"Beschreibe den Stiel des Pilzes: {stiel}\n"
        f"Beschreibe, ob der Pilz Lamellen oder einen Schwamm hat: {lamellen}\n"
        f"Beschreibe, ob der Pilz essbar ist: {essbar}\n\n"
        "=== Antwort (nur auf Deutsch, vollständige Sätze, ausschließlich aus dem Kontext): ==="
    )

    beschreibung = frage_mit_ollama(beschreibung_prompt)

    return {"klassifikation": klassifikations_info, "pilzname": PILZ_NAME, "beschreibung": beschreibung}

# === Frage Antwort ===
def beantworte_frage_mit_slm(frage):
    if not PILZ_NAME or not kontext:
        return "Bitte zuerst ein Bild hochladen und analysieren. (Oder setze_test_pilz() für Tests verwenden.)"

    prompt = (
        f"Es geht um den Pilz: {PILZ_NAME}\n\n"
        "Bilde einen Fließtext aus vollständigen deutschen Sätzen nur anhand folgender Informationen.\n"
        "Weitere Informationen dürfen nicht hinzugefügt werden.\n\n"
        f"=== Kontext ===\n{kontext}\n\n"
        "Folgende Frage soll anhand der deutschen Sätze präzise und direkt beantwortet werden:\n\n"
        f"=== Frage ===\n{frage}\n\n"
        "=== Antwort (ausschließlich auf Deutsch, vollständige Sätze): ==="
    )
    return frage_mit_ollama(prompt)

# === GUI ===
class PilzGUI:
    def __init__(self, master):
        self.master = master
        master.title("🍄 Pilz-Experte")
        master.geometry("1300x850")
        self.bg_color = "#1e1e1e"
        self.fg_color = "#f5f5f5"
        self.text_bg = "#2b2b2b"
        self.text_fg = "#f5f5f5"
        master.configure(bg=self.bg_color)

        self.title_label = tk.Label(master, text="🍄 Pilz-Experte", font=("Arial", 22, "bold"), bg=self.bg_color, fg=self.fg_color)
        self.title_label.pack(pady=10)

        main_frame = tk.Frame(master, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 5))

        self.chat_box = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=70, bg=self.text_bg, fg=self.text_fg, font=("Arial", 12), state="disabled")
        self.chat_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10), rowspan=3)

        self.bild_label = tk.Label(main_frame, text="Kein Bild hochgeladen", bg="#2a2a2a", fg="#aaaaaa", font=("Arial", 14, "italic"), width=60, height=28, relief="ridge", bd=1, anchor="center")
        self.bild_label.grid(row=0, column=1, sticky="n", padx=10, pady=(0, 5))

        self.pfad_label = tk.Label(main_frame, text="", bg=self.bg_color, fg="#888888", font=("Arial", 10, "italic"), wraplength=500, justify="center")
        self.pfad_label.grid(row=1, column=1, pady=(0, 5))

        self.upload_button = tk.Button(main_frame, text="📷 Bild hochladen", command=self.bild_auswaehlen, font=("Arial", 12), relief="groove", width=25, bg="#333333", fg=self.fg_color, activebackground="#444444")
        self.upload_button.grid(row=2, column=1, pady=(5, 0))

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        input_frame = tk.Frame(master, bg=self.bg_color)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

        self.entry = tk.Entry(input_frame, font=("Arial", 12), bg=self.text_bg, fg=self.text_fg, state="disabled")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.send_button = tk.Button(input_frame, text="Senden", command=self.sende_text, font=("Arial", 12), relief="groove", bg="#333333", fg=self.fg_color, activebackground="#444444", state="disabled")
        self.send_button.pack(side=tk.RIGHT)

        self.placeholder_text = "Stelle eine Frage..."
        self.entry_has_placeholder = False
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<FocusOut>", self._show_placeholder)

        self.bild_pfad = None

    def _show_placeholder(self, event=None):
        if self.entry.get().strip() == "":
            self.entry_has_placeholder = True
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self.placeholder_text)
            self.entry.config(fg="#888888")

    def _clear_placeholder(self, event=None):
        if self.entry_has_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg=self.text_fg)
            self.entry_has_placeholder = False

    def bild_auswaehlen(self):
        bild_pfad = filedialog.askopenfilename(title="Bild auswählen", filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not bild_pfad:
            return
        try:
            img = Image.open(bild_pfad)
            img.thumbnail((600, 600))
            img_tk = ImageTk.PhotoImage(img)
            self.bild_label.configure(image=img_tk, text="", width=600, height=600)
            self.bild_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")
            return

        self.pfad_label.config(text=f"Pfad: {os.path.abspath(bild_pfad)}")
        self.entry.config(state="normal")
        self.send_button.config(state="normal")
        self._show_placeholder()
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, "✅ Bild erfolgreich hochgeladen.\n")
        self.chat_box.insert(tk.END, "🔍 Bild wird analysiert...\n\n")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

        self.bild_pfad = bild_pfad
        threading.Thread(target=self.analyse_bild, args=(bild_pfad,)).start()

    def analyse_bild(self, bild_pfad):
        try:
            result = initialisiere_rag_mit_bild(bild_pfad)
            info = f"{result['klassifikation']}\n\n{result['beschreibung']}\n"
            self.chat_box.config(state="normal")
            self.chat_box.insert(tk.END, info + "\n")
            self.chat_box.config(state="disabled")
            self.chat_box.see(tk.END)
        except Exception as e:
            self.chat_box.config(state="normal")
            self.chat_box.insert(tk.END, f"❌ Fehler bei der Bildanalyse: {e}\n")
            self.chat_box.config(state="disabled")
            self.chat_box.see(tk.END)

    def sende_text(self):
        user_text = self.entry.get().strip()
        if not user_text or self.entry_has_placeholder:
            return
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, f"🧑‍💻 Du: {user_text}\n")
        self.chat_box.insert(tk.END, f"🍄 Pilz-Experte: (Antwort folgt...)\n\n")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)
        self.entry.delete(0, tk.END)
        self._show_placeholder()
        threading.Thread(target=self.get_answer, args=(user_text,)).start()

    def get_answer(self, frage):
        try:
            antwort = beantworte_frage_mit_slm(frage)
            self.chat_box.config(state="normal")
            self.chat_box.insert(tk.END, f"🍄 Pilz-Experte: {antwort}\n\n")
            self.chat_box.config(state="disabled")
            self.chat_box.see(tk.END)
        except Exception as e:
            self.chat_box.config(state="normal")
            self.chat_box.insert(tk.END, f"❌ Fehler: {e}\n")
            self.chat_box.config(state="disabled")
            self.chat_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = PilzGUI(root)
    root.update()
    threading.Thread(target=lade_embeddings_async, daemon=True).start()
    root.mainloop()