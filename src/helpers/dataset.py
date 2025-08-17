import kagglehub
import os
import shutil
from pathlib import Path

# Zielordner definieren
target_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Webscraper", "data", "images_mushrooms")
target_dir = os.path.abspath(target_dir)

print(f"Zielverzeichnis: {target_dir}")

# Zielordner erstellen falls nicht vorhanden
os.makedirs(target_dir, exist_ok=True)

# Download latest version
print("📥 Lade Kaggle Mushroom Dataset herunter...")
path = kagglehub.dataset_download("derekkunowilliams/mushrooms")

print("Path to dataset files:", path)

# Alle PNG-Dateien finden und kopieren
print("🔍 Suche nach PNG-Dateien...")
source_path = Path(path)
png_files = list(source_path.rglob("*.png"))

print(f"Gefunden: {len(png_files)} PNG-Dateien")

if png_files:
    print(f"📂 Kopiere Dateien nach {target_dir}...")
    
    for png_file in png_files:
        # Relativen Pfad beibehalten für Ordnerstruktur
        rel_path = png_file.relative_to(source_path)
        target_file = os.path.join(target_dir, rel_path)
        
        # Zielordner erstellen
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Datei kopieren
        shutil.copy2(png_file, target_file)
        print(f"✅ Kopiert: {rel_path}")
    
    print(f"🎉 Fertig! {len(png_files)} PNG-Dateien nach {target_dir} kopiert.")
else:
    print("❌ Keine PNG-Dateien im Dataset gefunden.")
    print("Dataset-Inhalt:")
    for item in source_path.rglob("*"):
        if item.is_file():
            print(f"  - {item.name} ({item.suffix})")