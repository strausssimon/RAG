
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

# === Globale Variablen für GUI-Kommunikation ===
PILZ_NAME = None
klassifikations_info = ""
vorstellung_text = ""
kontext = ""
pilz_info = None

def initialisiere_rag_mit_bild(image_path=None):
    """
    Initialisiert das RAG-System mit einem gegebenen Bildpfad.
    Gibt ein Dict mit Klassifikation, Pilzname, Vorstellungstext und Kontext zurück.
    """
    global PILZ_NAME, klassifikations_info, vorstellung_text, kontext, pilz_info

    # CNN Modellpfad bestimmen
    model_path = CNN_MODEL_PATH

    # Bildpfad bestimmen
    if image_path is None:
        # Fallback: Suche erstes Bild im RAG-Ordner
        image_path = finde_erstes_bild(os.path.dirname(__file__))

    # Klassifikation durchführen
    if image_path and os.path.exists(image_path):
        try:
            klasse, score = klassifiziere_pilzbild(image_path, model_path)
            PILZ_NAME = klasse
            klassifikations_info = f"Das analysierte Bild wurde mit {score:.1%} Wahrscheinlichkeit als {klasse} klassifiziert. "
        except Exception as e:
            PILZ_NAME = "Gemeiner Steinpilz"
            klassifikations_info = f"Bildklassifikation fehlgeschlagen ({e}). Verwende Fallback-Pilz. "
    else:
        PILZ_NAME = "Gemeiner Steinpilz"
        klassifikations_info = "Kein Bild zur Analyse gefunden. Verwende Fallback-Pilz. "

    # Pilzdaten laden
    with open(PILZ_DATEI, "r", encoding="utf-8") as f:
        pilzdaten = json.load(f)

    # Pilzinfo suchen
    pilz_info = next((p for p in pilzdaten if p["bezeichnung"]["name"] == PILZ_NAME), None)
    if pilz_info is None:
        raise ValueError(f"Pilz '{PILZ_NAME}' nicht in der JSON-Datei gefunden!")

    # Vorstellungstext generieren
    hut = pilz_info["aussehen"].get("hut", "unbekannt")
    stiel = pilz_info["aussehen"].get("stiel", "unbekannt")
    lamellen = pilz_info["aussehen"].get("schwamm_lamellen", "unbekannt")
    essbar = pilz_info["verzehr"].get("essbar", "unbekannt")
    vorstellung_text = (
        f"Der Pilz ist der {PILZ_NAME}. "
        f"Sein Hut ist {hut}, der Stiel ist {stiel} und er hat {lamellen}. "
        f"Hinsichtlich der Essbarkeit gilt: {essbar}."
    )

    # Kontext als JSON
    kontext = json.dumps(pilz_info, ensure_ascii=False, indent=2)

    return {
        "klassifikation": klassifikations_info,
        "pilzname": PILZ_NAME,
        "vorstellung": vorstellung_text,
        "kontext": kontext
    }

def beantworte_frage_mit_slm(frage):
    """
    Beantwortet eine Nutzerfrage mit dem SLM, basierend auf dem aktuellen Kontext.
    """
    if not PILZ_NAME or not kontext:
        return "Bitte zuerst ein Bild hochladen und klassifizieren."
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

# Optional: Interaktive Schleife absichern
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, messagebox
    from PIL import Image, ImageTk
    import os
    import threading

    class PilzGUI:
        def __init__(self, master):
            self.master = master
            master.title("🍄 Pilz-Experte")
            master.geometry("1300x850")

            # --- Dark Mode Farben ---
            self.bg_color = "#1e1e1e"
            self.fg_color = "#f5f5f5"
            self.text_bg = "#2b2b2b"
            self.text_fg = "#f5f5f5"

            master.configure(bg=self.bg_color)

            # --- Titel ---
            self.title_label = tk.Label(
                master, text="🍄 Pilz-Experte",
                font=("Arial", 22, "bold"),
                bg=self.bg_color, fg=self.fg_color
            )
            self.title_label.pack(pady=10)

            # --- Hauptbereich (Chat links + Bild rechts) ---
            main_frame = tk.Frame(master, bg=self.bg_color)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 5))

            # Chatbereich (links)
            self.chat_box = scrolledtext.ScrolledText(
                main_frame, wrap=tk.WORD, width=70,
                bg=self.text_bg, fg=self.text_fg, font=("Arial", 12),
                state="disabled"
            )
            self.chat_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10), rowspan=3)

            # Bildbereich (rechts oben)
            self.bild_label = tk.Label(
                main_frame, text="Kein Bild hochgeladen",
                bg="#2a2a2a", fg="#aaaaaa",
                font=("Arial", 14, "italic"), width=60, height=28,
                relief="ridge", bd=1, anchor="center"
            )
            self.bild_label.grid(row=0, column=1, sticky="n", padx=10, pady=(0, 5))

            # Pfad-Anzeige unter dem Bild
            self.pfad_label = tk.Label(
                main_frame, text="", bg=self.bg_color, fg="#888888",
                font=("Arial", 10, "italic"), wraplength=500, justify="center"
            )
            self.pfad_label.grid(row=1, column=1, pady=(0, 5))

            # Button unter Bild
            self.upload_button = tk.Button(
                main_frame, text="📷 Bild hochladen", command=self.bild_auswaehlen,
                font=("Arial", 12), relief="groove", width=25,
                bg="#333333", fg=self.fg_color, activebackground="#444444"
            )
            self.upload_button.grid(row=2, column=1, pady=(5, 0))

            # Grid anpassen
            main_frame.columnconfigure(0, weight=3)
            main_frame.columnconfigure(1, weight=2)
            main_frame.rowconfigure(0, weight=1)

            # --- Eingabefeld + Button (fix unten) ---
            input_frame = tk.Frame(master, bg=self.bg_color)
            input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

            self.entry = tk.Entry(input_frame, font=("Arial", 12),
                                  bg=self.text_bg, fg=self.text_fg, state="disabled")
            self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

            self.send_button = tk.Button(
                input_frame, text="Senden", command=self.sende_text,
                font=("Arial", 12), relief="groove",
                bg="#333333", fg=self.fg_color, activebackground="#444444",
                state="disabled"
            )
            self.send_button.pack(side=tk.RIGHT)

            # Placeholder Text
            self.placeholder_text = "Stelle eine Frage..."
            self.entry_has_placeholder = False

            # Fokus-Events für Placeholder
            self.entry.bind("<FocusIn>", self._clear_placeholder)
            self.entry.bind("<FocusOut>", self._show_placeholder)

            # Bildpfad für Analyse
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
            bild_pfad = filedialog.askopenfilename(
                title="Bild auswählen",
                filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            if not bild_pfad:
                return

            # Bild anzeigen (rechts)
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
            self.chat_box.insert(tk.END, f"✅ Bild erfolgreich hochgeladen.\n")
            self.chat_box.insert(tk.END, "🔍 Bild wird analysiert...\n\n")
            self.chat_box.config(state="disabled")
            self.chat_box.see(tk.END)

            self.bild_pfad = bild_pfad
            threading.Thread(target=self.analyse_bild, args=(bild_pfad,)).start()

        def analyse_bild(self, bild_pfad):
            try:
                result = initialisiere_rag_mit_bild(bild_pfad)
                info = f"{result['klassifikation']}\n\n{result['vorstellung']}\n"
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

    root = tk.Tk()
    app = PilzGUI(root)
    root.mainloop()
