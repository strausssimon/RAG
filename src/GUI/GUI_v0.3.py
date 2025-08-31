"""
====================================================
Programmname : GUI: Pilz-Experte Version 0.3
Beschreibung : GUI-Anwendung zur Interaktion mit RAG für die Pilzidentifikation.

====================================================
"""
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

class PilzGUI:
    def __init__(self, master):
        self.master = master
        master.title("🍄 Pilz-Experte")
        master.geometry("1200x800")
        master.configure(bg="white")

        # Titel
        self.title_label = tk.Label(master, text="🍄 Pilz-Experte", font=("Arial", 22, "bold"),
                                    bg="white", fg="black")
        self.title_label.pack(pady=10)

        # Hauptbereich: Text links, Bild rechts
        main_frame = tk.Frame(master, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # Textfeld (links)
        self.textfeld = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=60, height=30,
            font=("Arial", 12), bg="white", fg="black", relief="solid", bd=1
        )
        self.textfeld.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        # Bildbereich (rechts)
        self.bild_label = tk.Label(
            main_frame, text="Kein Bild geladen", bg="#f0f0f0", fg="gray",
            width=60, height=28, relief="ridge", bd=2, anchor="center"
        )
        self.bild_label.grid(row=0, column=1, sticky="n")

        # Grid-Konfiguration für saubere Verteilung
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # Button-Leiste unten
        button_frame = tk.Frame(master, bg="white")
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        self.upload_button = tk.Button(
            button_frame, text="📷 Bild hochladen", command=self.bild_auswaehlen,
            font=("Arial", 12), bg="#e0e0e0", fg="black", relief="raised", width=20
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.senden_button = tk.Button(
            button_frame, text="Senden", command=self.senden_text,
            font=("Arial", 12), bg="#d0d0d0", fg="black", relief="raised", width=15
        )
        self.senden_button.pack(side=tk.RIGHT, padx=10)

    def bild_auswaehlen(self):
        dateipfad = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not dateipfad:
            return

        try:
            bild = Image.open(dateipfad)
            bild.thumbnail((600, 600))
            self.bild_tk = ImageTk.PhotoImage(bild)
            self.bild_label.configure(image=self.bild_tk, text="")
            self.bild_label.image = self.bild_tk  # Referenz speichern
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")

    def senden_text(self):
        inhalt = self.textfeld.get("1.0", tk.END).strip()
        if not inhalt:
            messagebox.showinfo("Hinweis", "Bitte gib einen Text ein.")
        else:
            print("Eingabe:", inhalt)
            self.textfeld.insert(tk.END, f"\n🍄 Antwort: (Simulation...)\n")

# Start der Anwendung
if __name__ == "__main__":
    root = tk.Tk()
    app = PilzGUI(root)
    root.mainloop()