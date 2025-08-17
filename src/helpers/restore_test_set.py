import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm

def restore_test_images_to_classes(test_dir="Webscraper/data/test_mushrooms",
                                   target_dir="Webscraper/data/resized_mushrooms"):
    """
    Verschiebt Bilder aus dem Test-Ordner zurück in ihre ursprünglichen Klassenordner
    basierend auf den Dateinamen
    """
    test_path = Path(test_dir)
    target_path = Path(target_dir)
    
    if not test_path.exists():
        print(f"Fehler: Test-Ordner {test_path} existiert nicht!")
        return
    
    print(f"Verschiebe Test-Bilder zurück in Klassenordner...")
    print("=" * 60)
    
    # Alle Bilddateien im Test-Ordner finden
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(list(test_path.glob(f"*{ext}")))
        test_images.extend(list(test_path.glob(f"*{ext.upper()}")))
    
    if not test_images:
        print("Keine Test-Bilder gefunden!")
        return
    
    print(f"Gefunden: {len(test_images)} Test-Bilder")
    
    # Klassennamen aus Dateinamen extrahieren
    class_stats = {}
    moved_count = 0
    failed_count = 0
    
    # Bekannte Klassenordner sicherstellen
    known_classes = ["Amanita_phalloides", "Armillaria_mellea", "Boletus_edulis", "Cantharellus_cibarius"]
    for class_name in known_classes:
        class_dir = target_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVerschiebe Bilder zurück in Klassenordner...")
    
    for img_file in tqdm(test_images, desc="Verschiebe Dateien zurück"):
        try:
            filename = img_file.name
            
            # Klassenname aus Dateiname extrahieren
            detected_class = None
            for class_name in known_classes:
                if filename.startswith(class_name):
                    detected_class = class_name
                    break
            
            if detected_class is None:
                print(f"   Warnung: Kann Klasse für {filename} nicht bestimmen")
                failed_count += 1
                continue
            
            # Zielordner für diese Klasse
            target_class_dir = target_path / detected_class
            target_file = target_class_dir / filename
            
            # Falls Datei bereits existiert, eindeutigen Namen erstellen
            counter = 1
            original_target = target_file
            while target_file.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_file = target_class_dir / f"{stem}_restored_{counter}{suffix}"
                counter += 1
            
            # Datei verschieben
            shutil.move(str(img_file), str(target_file))
            
            # Statistiken aktualisieren
            if detected_class not in class_stats:
                class_stats[detected_class] = 0
            class_stats[detected_class] += 1
            moved_count += 1
            
        except Exception as e:
            print(f"   Fehler bei {img_file.name}: {e}")
            failed_count += 1
    
    # Ergebnisse anzeigen
    print(f"\n" + "=" * 60)
    print("WIEDERHERSTELLUNG ABGESCHLOSSEN")
    print("=" * 60)
    print(f"✅ Erfolgreich zurück verschoben: {moved_count} Dateien")
    print(f"❌ Fehlgeschlagen: {failed_count} Dateien")
    
    print(f"\nBILDER ZURÜCK IN KLASSEN:")
    for class_name, count in sorted(class_stats.items()):
        print(f"   {class_name}: {count} Bilder")
    
    # Test-Ordner löschen wenn leer
    remaining_files = list(test_path.glob("*"))
    if not remaining_files:
        try:
            test_path.rmdir()
            print(f"\n🗑️ Leerer Test-Ordner entfernt: {test_path}")
        except:
            print(f"\n⚠️ Konnte Test-Ordner nicht entfernen: {test_path}")
    else:
        print(f"\n📁 Test-Ordner enthält noch {len(remaining_files)} Dateien")
    
    return moved_count, failed_count, class_stats

def clean_test_directory(test_dir="Webscraper/data/test_mushrooms"):
    """
    Entfernt alle Dateien aus dem Test-Ordner (falls man komplett neu starten möchte)
    """
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"Test-Ordner {test_path} existiert nicht!")
        return
    
    # Alle Dateien finden
    all_files = list(test_path.glob("*"))
    
    if not all_files:
        print(f"Test-Ordner {test_path} ist bereits leer!")
        return
    
    print(f"WARNUNG: Lösche {len(all_files)} Dateien aus {test_path}")
    choice = input("Sind Sie sicher? (j/n): ").lower().strip()
    
    if choice in ['j', 'ja', 'y', 'yes']:
        deleted_count = 0
        for file_path in tqdm(all_files, desc="Lösche Dateien"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    deleted_count += 1
            except Exception as e:
                print(f"Fehler beim Löschen von {file_path}: {e}")
        
        print(f"✅ {deleted_count} Dateien/Ordner gelöscht")
        
        # Versuche Test-Ordner zu entfernen
        try:
            test_path.rmdir()
            print(f"🗑️ Test-Ordner entfernt: {test_path}")
        except:
            print(f"⚠️ Konnte Test-Ordner nicht entfernen (nicht leer?)")
    else:
        print("Löschvorgang abgebrochen")

if __name__ == "__main__":
    print("🔄 TEST-SET MANAGER - WIEDERHERSTELLUNG 🔄")
    print("=" * 60)
    
    print("Optionen:")
    print("1. Test-Bilder zurück in Klassenordner verschieben")
    print("2. Test-Ordner komplett leeren (VORSICHT!)")
    print("3. Abbrechen")
    
    choice = input("\nWählen Sie eine Option (1/2/3): ").strip()
    
    if choice == "1":
        print("\n" + "="*50)
        print("Verschiebe Test-Bilder zurück in ihre Klassenordner...")
        moved, failed, stats = restore_test_images_to_classes()
        
        if moved > 0:
            print(f"\n✅ {moved} Bilder erfolgreich zurück verschoben!")
            print("Sie können jetzt das korrigierte create_test_set.py ausführen.")
        else:
            print("\n❌ Keine Bilder verschoben!")
            
    elif choice == "2":
        print("\n" + "="*50)
        clean_test_directory()
        
    else:
        print("Vorgang abgebrochen.")
