"""
====================================================
Programmname : GUI: Pilz-Experte 
Beschreibung : GUI-Anwendung zur Interaktion mit RAG für die Pilzidentifikation.

====================================================
"""
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import os

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

    def _show_placeholder(self, event=None):
        """Zeigt den Placeholder, wenn das Feld leer ist"""
        if self.entry.get().strip() == "":
            self.entry_has_placeholder = True
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self.placeholder_text)
            self.entry.config(fg="#888888")

    def _clear_placeholder(self, event=None):
        """Entfernt den Placeholder, wenn der Nutzer tippt"""
        if self.entry_has_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg=self.text_fg)
            self.entry_has_placeholder = False

    def bild_auswaehlen(self):
        """Öffnet einen Dateidialog zum Hochladen eines Bildes"""
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

        # Pfad unter dem Bild anzeigen
        self.pfad_label.config(text=f"Pfad: {os.path.abspath(bild_pfad)}")

        # Eingabe aktivieren
        self.entry.config(state="normal")
        self.send_button.config(state="normal")

        # Placeholder anzeigen
        self._show_placeholder()

        # Chatbox Meldungen
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, f"✅ Bild erfolgreich hochgeladen.\n")
        self.chat_box.insert(tk.END, "🔍 Bild wird analysiert...\n\n")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

    def sende_text(self):
        """Fügt die Eingabe in die Chatbox ein"""
        user_text = self.entry.get().strip()
        if not user_text or self.entry_has_placeholder:
            return

        # In Chatbox schreiben
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, f"🧑‍💻 Du: {user_text}\n")
        self.chat_box.insert(tk.END, f"🍄 Pilz-Experte: (Antwort folgt...)\n\n")
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)

        # Eingabefeld leeren und Placeholder wieder anzeigen
        self.entry.delete(0, tk.END)
        self._show_placeholder()


if __name__ == "__main__":
    root = tk.Tk()
    app = PilzGUI(root)
    root.mainloop()