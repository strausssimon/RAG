import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class PilzGUI:
    def __init__(self, master):
        self.master = master
        master.title("Pilz-Experte")
        master.geometry("1000x700")
        master.configure(bg="white")

        # Titel
        self.title_label = tk.Label(master, text="🍄 Pilz-Experte", font=("Arial", 22, "bold"),
                                    bg="white", fg="black")
        self.title_label.pack(pady=10)

        # Hauptbereich: Text (links) und Bild (rechts)
        main_frame = tk.Frame(master, bg="white")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Textfeld (links)
        self.textfeld = tk.Text(main_frame, wrap=tk.WORD, width=40, height=20,
                                font=("Arial", 12), bg="white", fg="black", relief="solid", bd=1)
        self.textfeld.grid(row=0, column=0, padx=(0, 20), sticky="n")

        # Bildanzeige (rechts)
        self.bild_label = tk.Label(main_frame, text="Kein Bild geladen", bg="#f0f0f0", fg="gray",
                                   width=50, height=20, relief="ridge", bd=2)
        self.bild_label.grid(row=0, column=1, sticky="n")

        # Button-Bereich unten
        button_frame = tk.Frame(master, bg="white")
        button_frame.pack(pady=10)

        self.upload_button = tk.Button(button_frame, text="📷 Bild hochladen", command=self.bild_auswaehlen,
                                       font=("Arial", 12), bg="#e0e0e0", fg="black")
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.senden_button = tk.Button(button_frame, text="Senden", command=self.senden,
                                       font=("Arial", 12), bg="#d0d0d0", fg="black")
        self.senden_button.pack(side=tk.LEFT)

    def bild_auswaehlen(self):
        dateipfad = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilddateien", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not dateipfad:
            return

        try:
            bild = Image.open(dateipfad)
            bild.thumbnail((400, 400))
            self.bild_tk = ImageTk.PhotoImage(bild)
            self.bild_label.configure(image=self.bild_tk, text="")
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{e}")

    def senden(self):
        inhalt = self.textfeld.get("1.0", tk.END).strip()
        if not inhalt:
            messagebox.showinfo("Hinweis", "Bitte gib einen Text ein.")
        else:
            print("Eingegebener Text:", inhalt)  # Beispielverhalten – hier könnte KI-Logik rein

# Start
if __name__ == "__main__":
    root = tk.Tk()
    app = PilzGUI(root)
    root.mainloop()