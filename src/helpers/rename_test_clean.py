"""
====================================================
Programmname :  Rename Test Images
Datum        :  17.08.2025
Version      :  1.0
Beschreibung :  Benennt alle Bilder im gewählten Ordner um

====================================================
"""
import re
from pathlib import Path

def extract_mushroom_info(filename):
    """
    Extrahiert Pilzname und Nummer aus dem Dateinamen.
    Behandelt auch Duplikat-Suffixes wie _dup1, _copy1, etc.
    """
    # Entferne Dateiendung
    name_without_ext = filename.split('.')[0]
    
    # Entferne Duplikat-Suffixes wie _dup1, _copy1, _restored1, _backup1 etc.
    clean_name = re.sub(r'_(dup|copy|restored|backup)\d*$', '', name_without_ext, flags=re.IGNORECASE)
    
    # Bekannte Pilzklassen (lowercase für Vergleich)
    known_mushrooms = [
        'amanita_muscaria',
        'amanita_pantherina', 
        'amanita_phalloides',
        'armillaria_mellea',
        'boletus_edulis',
        'cantharellus_cibarius',
        'imleria_badia',
        'tylopilus_felleus'
    ]
    
    # Versuche bekannte Pilznamen zu finden
    for mushroom in known_mushrooms:
        if clean_name.lower().startswith(mushroom):
            # Extrahiere die Nummer nach dem Pilznamen
            remaining = clean_name[len(mushroom):]
            # Entferne führende Unterstriche oder andere Zeichen
            remaining = remaining.lstrip('_')
            
            # Prüfe ob Rest eine Nummer ist
            if remaining.isdigit():
                return mushroom, remaining
    
    # Fallback: Suche nach Muster [text]_[nummer]
    match = re.match(r'^(.+?)_?(\d+)$', clean_name)
    if match:
        text_part, number_part = match.groups()
        # Normalisiere den Textteil (lowercase, ersetze Leerzeichen mit _)
        normalized_text = text_part.lower().replace(' ', '_')
        return normalized_text, number_part
    
    # Wenn keine Nummer gefunden, return den Namen und None
    return clean_name.lower(), None

def rename_test_images():
    """
    Benennt alle Bilder im test_mushrooms Ordner um zu standardisiertem Format:
    pilzname_nummer.extension (alles lowercase)
    Entfernt Duplikat-Suffixes wie _dup1, _copy1, etc.
    """
    test_path = Path("Webscraper/data/test_mushrooms")
    
    if not test_path.exists():
        print(f"Fehler: Ordner {test_path} existiert nicht!")
        return
    
    # Alle Bilddateien finden
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    all_images = []
    
    for file_path in test_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            all_images.append(file_path)
    
    if not all_images:
        print("Keine Bilddateien gefunden!")
        return
    
    print(f"Gefunden: {len(all_images)} Bilddateien")
    print("Beginne Umbenennung...\n")
    
    renamed_count = 0
    errors = []
    
    for img_file in all_images:
        old_name = img_file.name
        file_extension = img_file.suffix.lower()
        
        # Extrahiere Pilzname und Nummer
        mushroom_name, number = extract_mushroom_info(old_name)
        
        if number is None:
            # Falls keine Nummer erkennbar, verwende laufende Nummer
            number = str(renamed_count + 1)
            print(f"⚠️  Keine Nummer erkennbar in '{old_name}', verwende: {number}")
        
        # Neuen Namen erstellen: pilzname_nummer.extension (alles lowercase)
        new_name = f"{mushroom_name}_{number}{file_extension}"
        
        # Prüfen ob Umbenennung nötig ist
        if old_name == new_name:
            print(f"✓ '{old_name}' bereits korrekt benannt")
            continue
        
        new_path = img_file.parent / new_name
        
        # Prüfen ob Zieldatei bereits existiert
        if new_path.exists():
            print(f"⚠️  Zieldatei '{new_name}' existiert bereits, überspringe '{old_name}'")
            continue
        
        try:
            # Datei umbenennen
            img_file.rename(new_path)
            print(f"✓ '{old_name}' → '{new_name}'")
            renamed_count += 1
            
        except Exception as e:
            error_msg = f"Fehler beim Umbenennen von '{old_name}': {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
    
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG:")
    print(f"Insgesamt {len(all_images)} Dateien überprüft")
    print(f"Erfolgreich umbenannt: {renamed_count}")
    
    if errors:
        print(f"Fehler: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    print("Umbenennung abgeschlossen!")

def preview_renaming():
    """
    Zeigt eine Vorschau der geplanten Umbenennungen an, ohne sie durchzuführen.
    """
    test_path = Path("Webscraper/data/test_mushrooms")
    
    if not test_path.exists():
        print(f"Fehler: Ordner {test_path} existiert nicht!")
        return
    
    # Alle Bilddateien finden
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    all_images = []
    
    for file_path in test_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            all_images.append(file_path)
    
    if not all_images:
        print("Keine Bilddateien gefunden!")
        return
    
    print(f"VORSCHAU: {len(all_images)} Bilddateien werden überprüft\n")
    
    changes_needed = 0
    
    for img_file in all_images:
        old_name = img_file.name
        file_extension = img_file.suffix.lower()
        
        # Extrahiere Pilzname und Nummer
        mushroom_name, number = extract_mushroom_info(old_name)
        
        if number is None:
            number = "X"  # Placeholder für Vorschau
        
        # Neuen Namen erstellen
        new_name = f"{mushroom_name}_{number}{file_extension}"
        
        if old_name != new_name:
            print(f"'{old_name}' → '{new_name}'")
            changes_needed += 1
        else:
            print(f"'{old_name}' (bereits korrekt)")
    
    print(f"\nVorschau: {changes_needed} von {len(all_images)} Dateien würden umbenannt werden.")

def main():
    """Hauptfunktion mit Benutzerinteraktion"""
    print("Datei-Umbenennung für Test-Mushroom-Dataset")
    print("=" * 50)
    print("1. Vorschau anzeigen")
    print("2. Umbenennung durchführen")
    print("3. Beenden")
    
    while True:
        choice = input("\nWählen Sie eine Option (1-3): ").strip()
        
        if choice == "1":
            preview_renaming()
        elif choice == "2":
            confirm = input("Möchten Sie die Umbenennung wirklich durchführen? (j/n): ").strip().lower()
            if confirm in ['j', 'ja', 'y', 'yes']:
                rename_test_images()
            else:
                print("Umbenennung abgebrochen.")
        elif choice == "3":
            print("Programm beendet.")
            break
        else:
            print("Ungültige Eingabe. Bitte wählen Sie 1, 2 oder 3.")

if __name__ == "__main__":
    main()
