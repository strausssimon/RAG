"""
====================================================
Programmname :  Randomize and Move Images
Beschreibung :  Zufälliges Auswählen und Verschieben von Bildern in neue Verzeichnisse

====================================================
"""
import os
import shutil
import random

# Konfiguration - absolute Pfade verwenden
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
SOURCE_BASE = os.path.join(project_root, "data", "resized_mushrooms", "inaturalist")
TARGET_BASE = os.path.join(project_root, "data", "randomized_mushrooms", "inaturalist")
CLASSES = [
    "Amanita_muscaria",
    "Boletus_edulis",
    "Armillaria_mellea",
    "phallus_impudicus",
    "cantharellus_cibarius",
    ]
N_IMAGES = 1500

print(f"MUSHROOM RANDOMIZER - Kopiere {N_IMAGES} Bilder pro Klasse")
print("=" * 70)
print(f"Quelle: {SOURCE_BASE}")
print(f"Ziel:   {TARGET_BASE}")
print(f"Klassen: {len(CLASSES)} → {', '.join(CLASSES)}")
print("=" * 70)

os.makedirs(TARGET_BASE, exist_ok=True)

total_copied = 0
for cls in CLASSES:
    src_dir = os.path.join(SOURCE_BASE, cls)
    tgt_dir = os.path.join(TARGET_BASE, cls)
    
    print(f"\n📁 Verarbeite: {cls}")
    
    # Zielordner leeren falls schon vorhanden
    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir, exist_ok=True)
    
    if not os.path.exists(src_dir):
        print(f"❌ Quellordner nicht gefunden: {src_dir}")
        continue
        
    # Nur Bilddateien
    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"))]
    
    print(f"   Verfügbare Bilder: {len(files)}")
    
    if len(files) < N_IMAGES:
        print(f"⚠️  Warnung: {src_dir} enthält nur {len(files)} Bilder, weniger als {N_IMAGES}!")
        selected = files
    else:
        selected = random.sample(files, N_IMAGES)
    
    print(f"   Ausgewählte Bilder: {len(selected)}")
    
    # Kopieren mit Fortschritt
    copied_count = 0
    for fname in selected:
        src_path = os.path.join(src_dir, fname)
        tgt_path = os.path.join(tgt_dir, fname)
        try:
            shutil.copy2(src_path, tgt_path)
            copied_count += 1
        except Exception as e:
            print(f"   Fehler beim Kopieren {fname}: {e}")
    
    print(f"✅ {copied_count} Bilder erfolgreich kopiert")
    total_copied += copied_count

print("\n" + "=" * 70)
print(f"🎉 FERTIG: {total_copied} Bilder insgesamt kopiert")
print(f"💯 Erwartete Anzahl: {len(CLASSES)} × {N_IMAGES} = {len(CLASSES) * N_IMAGES}")
print(f"📊 Erfolgsrate: {total_copied}/{len(CLASSES) * N_IMAGES} = {(total_copied/(len(CLASSES) * N_IMAGES)*100):.1f}%")
print("=" * 70)
