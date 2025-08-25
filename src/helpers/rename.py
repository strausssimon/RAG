"""
====================================================
Programmname :  Rename Images
Beschreibung :  Benennt alle Bilder im gewählten Pfad um. 

====================================================
"""

import os
from pathlib import Path
import shutil

def rename_mushroom_images():
    """
    Benennt alle Bilder in den Unterordnern von Webscraper/data/images_mushrooms um.
    Format: [unterordner_name]_[ursprünglicher_name].jpg
    
    Beispiel: 
    - Webscraper/data/images_mushrooms/Amanita_phalloides/634.jpg 
      wird zu amanita_phalloides_634.jpg
    """
    
    # Basis-Pfad zu den Bildern
    base_path = Path("data/images_mushrooms")
    
    if not base_path.exists():
        print(f"Fehler: Ordner {base_path} existiert nicht!")
        return
    
    print(f"Starte Umbenennung in: {base_path.absolute()}")
    print("=" * 60)
    
    # Durch alle Unterordner iterieren
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            print(f"\nVerarbeite Ordner: {subfolder.name}")
            
            # Präfix aus dem Ordnernamen erstellen (lowercase)
            prefix = subfolder.name.lower()
            
            # Alle jpg-Dateien im Unterordner finden
            jpg_files = list(subfolder.glob("*.jpg"))
            
            if not jpg_files:
                print(f"   Keine .jpg Dateien in {subfolder.name} gefunden")
                continue
                
            print(f"   Gefunden: {len(jpg_files)} Bilder")
            
            renamed_count = 0
            
            # Jede jpg-Datei umbenennen
            for img_file in jpg_files:
                old_name = img_file.name
                
                # Neuen Namen erstellen: [prefix]_[alter_name]
                new_name = f"{prefix}_{old_name}"
                new_path = img_file.parent / new_name
                
                # Prüfen ob Datei bereits den gewünschten Namen hat
                if old_name.startswith(f"{prefix}_"):
                    print(f"   Überspringe {old_name} (bereits korrekt benannt)")
                    continue
                
                # Prüfen ob Zieldatei bereits existiert
                if new_path.exists():
                    print(f"   Überspringe {old_name} -> {new_name} (Ziel existiert bereits)")
                    continue
                
                try:
                    # Datei umbenennen
                    img_file.rename(new_path)
                    print(f"   {old_name} -> {new_name}")
                    renamed_count += 1
                    
                except Exception as e:
                    print(f"   Fehler beim Umbenennen von {old_name}: {e}")
            
            print(f"   {renamed_count} von {len(jpg_files)} Dateien umbenannt")
    
    print("\n" + "=" * 60)
    print("Umbenennung abgeschlossen!")

def preview_renaming():
    """
    Zeigt eine Vorschau der geplanten Umbenennungen an, ohne sie durchzuführen.
    """
    base_path = Path("data/resized_mushrooms/inaturalist")

    if not base_path.exists():
        print(f"Fehler: Ordner {base_path} existiert nicht!")
        return
    
    print(f"VORSCHAU - Geplante Umbenennungen in: {base_path.absolute()}")
    print("=" * 80)
    
    total_files = 0
    total_to_rename = 0
    
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            print(f"\nOrdner: {subfolder.name}")
            
            prefix = subfolder.name.lower()
            jpg_files = list(subfolder.glob("*.jpg"))
            
            if not jpg_files:
                print(f"   Keine .jpg Dateien gefunden")
                continue
            
            total_files += len(jpg_files)
            
            for img_file in jpg_files:
                old_name = img_file.name
                new_name = f"{prefix}_{old_name}"
                
                if old_name.startswith(f"{prefix}_"):
                    print(f"   {old_name} (bereits korrekt)")
                else:
                    print(f"   {old_name} -> {new_name}")
                    total_to_rename += 1
    
    print("\n" + "=" * 80)
    print(f"Zusammenfassung:")
    print(f"   Gesamt gefundene Dateien: {total_files}")
    print(f"   Dateien zum Umbenennen: {total_to_rename}")
    print(f"   Bereits korrekt benannt: {total_files - total_to_rename}")

if __name__ == "__main__":
    print("Mushroom Image Renamer")
    print("=" * 30)
    
    # Erst Vorschau anzeigen
    preview_renaming()
    
    # Benutzer fragen ob fortfahren
    print("\n" + "=" * 50)
    choice = input("Möchten Sie mit der Umbenennung fortfahren? (j/n): ").lower().strip()
    
    if choice in ['j', 'ja', 'y', 'yes']:
        print("\nStarte Umbenennung...")
        rename_mushroom_images()
    else:
        print("Umbenennung abgebrochen.")
