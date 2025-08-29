"""
====================================================
Programmname :  RobustTestSet
Beschreibung :  Überprüft und verschiebt Testbilder aus dem Quellverzeichnis in das Zielverzeichnis.

====================================================
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def check_and_move_test_images(source_dir="data/resized_mushrooms", 
                               target_dir="data/test_mushrooms", 
                               test_percentage=0.2):
    """
    Robuste Version: Überprüft zuerst vorhandene Dateien, dann verschiebt klassenweise
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Fehler: Quellverzeichnis {source_path} existiert nicht!")
        return
    
    # Zielverzeichnis erstellen
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Zielverzeichnis: {target_path.absolute()}")
    
    print(f"Robuste Verschiebung: {test_percentage*100:.0f}% pro Klasse")
    print("=" * 60)
    
    # Sammle tatsächlich vorhandene Dateien
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    print("1. Überprüfe tatsächlich vorhandene Dateien...")
    
    total_moved = 0
    total_failed = 0
    class_stats = {}
    
    random.seed(42)  # Reproduzierbare Ergebnisse
    
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\nVerarbeite Klasse: {class_name}")
        
        # Überprüfe welche Dateien wirklich existieren
        existing_files = []
        for img_file in class_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                existing_files.append(img_file)
        
        if not existing_files:
            print(f"   Keine Dateien in {class_name} gefunden")
            continue
            
        print(f"   Tatsächlich vorhanden: {len(existing_files)} Dateien")
        
        # Berechne 20% für Test
        test_count = int(len(existing_files) * test_percentage)
        if test_count == 0 and len(existing_files) > 0:
            test_count = 1  # Mindestens 1 Datei für Test
            
        print(f"   Für Test ausgewählt: {test_count} Dateien")
        
        # Zufällige Auswahl
        if test_count > 0:
            selected_files = random.sample(existing_files, min(test_count, len(existing_files)))
            
            class_moved = 0
            class_failed = 0
            
            for img_file in tqdm(selected_files, desc=f"Verschiebe {class_name}"):
                try:
                    # Prüfe nochmal ob Datei existiert
                    if not img_file.exists():
                        print(f"   Datei existiert nicht mehr: {img_file.name}")
                        class_failed += 1
                        continue
                    
                    # Zieldatei
                    target_file = target_path / img_file.name
                    
                    # Duplikat-Behandlung
                    counter = 1
                    original_target = target_file
                    while target_file.exists():
                        stem = original_target.stem
                        suffix = original_target.suffix
                        target_file = target_path / f"{stem}_copy{counter}{suffix}"
                        counter += 1
                    
                    # Verschieben
                    shutil.move(str(img_file), str(target_file))
                    class_moved += 1
                    
                except Exception as e:
                    print(f"   Fehler bei {img_file.name}: {e}")
                    class_failed += 1
            
            class_stats[class_name] = class_moved
            total_moved += class_moved
            total_failed += class_failed
            
            print(f"   Erfolgreich verschoben: {class_moved}/{test_count}")
            if class_failed > 0:
                print(f"   Fehlgeschlagen: {class_failed}")
    
    # Ergebnisse
    print(f"\n" + "=" * 60)
    print("ROBUSTE VERSCHIEBUNG ABGESCHLOSSEN")
    print("=" * 60)
    print(f"Gesamt erfolgreich: {total_moved} Dateien")
    print(f"Gesamt fehlgeschlagen: {total_failed} Dateien")
    
    if total_moved > 0:
        print(f"\nVERTEILUNG IM TEST-SET:")
        for class_name, count in sorted(class_stats.items()):
            percentage = (count / total_moved) * 100
            print(f"   {class_name}: {count} Bilder ({percentage:.1f}%)")
    
    return total_moved, total_failed, class_stats

def verify_current_state(source_dir="Webscraper/data/resized_mushrooms",
                        test_dir="Webscraper/data/test_mushrooms"):
    """
    Überprüft den aktuellen Zustand der Verzeichnisse
    """
    source_path = Path(source_dir)
    test_path = Path(test_dir)
    
    print("AKTUELLER ZUSTAND:")
    print("=" * 40)
    
    # Training (source) Verzeichnis
    total_training = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    print("Training-Bilder (verbleibend):")
    if source_path.exists():
        for class_dir in source_path.iterdir():
            if class_dir.is_dir():
                count = 0
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        count += 1
                total_training += count
                print(f"   {class_dir.name}: {count} Bilder")
    
    # Test Verzeichnis
    total_test = 0
    print("\nTest-Bilder:")
    if test_path.exists():
        for img_file in test_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                total_test += 1
        print(f"   Gesamt: {total_test} Bilder")
    else:
        print("   Test-Verzeichnis existiert nicht")
    
    total = total_training + total_test
    if total > 0:
        test_percentage = (total_test / total) * 100
        print(f"\nGESAMT-ÜBERSICHT:")
        print(f"   Training: {total_training} Bilder ({100-test_percentage:.1f}%)")
        print(f"   Test: {total_test} Bilder ({test_percentage:.1f}%)")
        print(f"   Gesamt: {total} Bilder")

def reset_test_split(source_dir="Webscraper/data/resized_mushrooms",
                     test_dir="Webscraper/data/test_mushrooms"):
    """
    Verschiebt alle Test-Bilder zurück in ihre ursprünglichen Ordner
    """
    test_path = Path(test_dir)
    source_path = Path(source_dir)
    
    if not test_path.exists():
        print("Kein Test-Verzeichnis vorhanden - nichts zu tun")
        return
    
    print("RESET: Verschiebe Test-Bilder zurück...")
    print("=" * 50)
    
    # Alle Test-Bilder finden
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    test_images = []
    for img_file in test_path.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            test_images.append(img_file)
    
    if not test_images:
        print("Keine Test-Bilder gefunden")
        return
    
    print(f"Gefunden: {len(test_images)} Test-Bilder")
    
    # Klassennamen aus Dateinamen extrahieren
    known_classes = ["amanita_phalloides", "armillaria_mellea", "boletus_edulis", "cantharellus_cibarius"]
    moved_back = 0
    failed = 0
    
    for img_file in tqdm(test_images, desc="Verschiebe zurück"):
        try:
            filename = img_file.name.lower()
            
            # Bestimme Klasse
            target_class = None
            for class_name in known_classes:
                if filename.startswith(class_name):
                    target_class = class_name
                    break
            
            if target_class is None:
                print(f"   Kann Klasse für {img_file.name} nicht bestimmen")
                failed += 1
                continue
            
            # Zielordner
            target_dir = source_path / target_class.title().replace("_", "_")  # Armillaria_mellea format
            if not target_dir.exists():
                # Versuche verschiedene Formate
                for class_dir in source_path.iterdir():
                    if class_dir.is_dir() and class_dir.name.lower().replace("_", "") == target_class.replace("_", ""):
                        target_dir = class_dir
                        break
            
            if not target_dir.exists():
                print(f"   Zielordner für {target_class} nicht gefunden")
                failed += 1
                continue
            
            # Verschieben
            target_file = target_dir / img_file.name
            counter = 1
            while target_file.exists():
                stem = img_file.stem
                suffix = img_file.suffix
                target_file = target_dir / f"{stem}_restored{counter}{suffix}"
                counter += 1
            
            shutil.move(str(img_file), str(target_file))
            moved_back += 1
            
        except Exception as e:
            print(f"   Fehler bei {img_file.name}: {e}")
            failed += 1
    
    print(f"\nZurück verschoben: {moved_back}")
    print(f"Fehlgeschlagen: {failed}")
    
    # Test-Ordner löschen wenn leer
    remaining = list(test_path.glob("*"))
    if not remaining:
        test_path.rmdir()
        print(f"Leerer Test-Ordner entfernt")

if __name__ == "__main__":
    print("ROBUSTER TEST-SET MANAGER")
    print("=" * 50)
    
    print("Optionen:")
    print("1. Aktuellen Zustand überprüfen")
    print("2. Reset: Test-Bilder zurück verschieben")
    print("3. Robuste 20% Test-Set Erstellung")
    print("4. Abbrechen")
    
    choice = input("\nWählen Sie eine Option (1-4): ").strip()
    
    if choice == "1":
        verify_current_state()
        
    elif choice == "2":
        print("\n" + "="*50)
        reset_test_split()
        print("\nNach Reset:")
        verify_current_state()
        
    elif choice == "3":
        print("\n" + "="*50)
        print("Starte robuste Test-Set Erstellung...")
        moved, failed, stats = check_and_move_test_images()
        
        if moved > 0:
            print(f"\n{moved} Bilder erfolgreich verschoben!")
            verify_current_state()
        
    else:
        print("Abgebrochen")
