"""
====================================================
Programmname : GUI: Pilz-Experte Version 0.1
Beschreibung : GUI-Anwendung zur Interaktion mit RAG für die Pilzidentifikation.

====================================================
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class WeissePilzGUI:
    def __init__(self, master):
        self.master = master
        master.title("Pilz-Experte")
        master.geometry("1000x700")
        master.configure(bg="white")

        # Titel
        self.title_label = tk.Label(master, text="🍄 Pilz-Experte", font=("Arial", 22, "bold"), bg="white", fg="black")
        self.title_label.pack(pady=10)

        # Bildanzeige
        self.bild_label = tk.Label(master, text="Kein Bild geladen", bg="#f0f0f0", fg="gray",
                                   width=80, height=25, relief="ridge", bd=2)
        self.bild_label.pack(pady=10)

        # Button zum Hochladen
        self.upload_button = tk.Button(master, text="Bild hochladen", command=self.bild_auswaehlen,
                                       font=("Arial", 12), bg="#e0e0e0", fg="black")
        self.upload_button.pack(pady=5)

        # Eingabefeld
        self.eingabe = tk.Entry(master, font=("Arial", 12), width=60, bg="white", fg="black", relief="solid", bd=1)
        self.eingabe.pack(pady=10)

    def bild_auswaehlen(self):
        dateipfad = filedialog.askopenfilename(title="Bild auswählen",
                                               filetypes=[("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not dateipfad:
            return

        try:
            bild = Image.open(dateipfad)
            bild.thumbnail((600, 500))
            self.bild_tk = ImageTk.PhotoImage(bild)
            self.bild_label.configure(image=self.bild_tk, text="")
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")

# Programm starten
if __name__ == "__main__":
    root = tk.Tk()
    app = WeissePilzGUI(root)
    root.mainloop()
