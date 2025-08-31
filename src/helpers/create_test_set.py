import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def move_random_test_images(source_dir="Webscraper/data/resized_mushrooms", 
                           target_dir="Webscraper/data/test_mushrooms", 
                           test_percentage=0.2):
    """
    Verschiebt randomisiert einen bestimmten Prozentsatz der Bilder 
    aus JEDER Klasse separat in einen gemeinsamen Test-Ordner
    
    Args:
        source_dir (str): Quellverzeichnis mit Unterordnern
        target_dir (str): Zielverzeichnis (flache Struktur)
        test_percentage (float): Prozentsatz für Test-Set (0.0 - 1.0)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Fehler: Quellverzeichnis {source_path} existiert nicht!")
        return
    
    # Zielverzeichnis erstellen
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Zielverzeichnis erstellt: {target_path.absolute()}")
    
    print(f"Verschiebt {test_percentage*100:.0f}% der Bilder aus JEDER Klasse nach {target_dir}")
    print("=" * 60)
    
    # Sammle Bilddateien pro Klasse
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    class_images = {}
    
    print("1. Sammle Bilddateien pro Klasse...")
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"   Analysiere Klasse: {class_name}")
        
        # Alle Bildformate finden
        images_in_class = []
        for ext in image_extensions:
            images_in_class.extend(list(class_dir.glob(f"*{ext}")))
            images_in_class.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        class_images[class_name] = images_in_class
        print(f"   Gefunden: {len(images_in_class)} Bilder")
    
    if not class_images:
        print("Keine Bilddateien gefunden!")
        return
    
    # Berechne Test-Anzahl pro Klasse
    print(f"\n2. Berechne {test_percentage*100:.0f}% Aufteilung pro Klasse:")
    selected_for_test = []
    class_test_counts = {}
    
    random.seed(42)  # Für reproduzierbare Ergebnisse
    
    for class_name, images in class_images.items():
        total_in_class = len(images)
        test_count_in_class = int(total_in_class * test_percentage)
        remaining_in_class = total_in_class - test_count_in_class
        
        print(f"   {class_name}:")
        print(f"      Gesamt: {total_in_class} Bilder")
        print(f"      Für Test: {test_count_in_class} Bilder ({test_percentage*100:.0f}%)")
        print(f"      Verbleiben: {remaining_in_class} Bilder")
        
        # Zufällige Auswahl aus dieser Klasse
        if test_count_in_class > 0:
            selected_from_class = random.sample(images, test_count_in_class)
            selected_for_test.extend(selected_from_class)
            class_test_counts[class_name] = test_count_in_class
        else:
            class_test_counts[class_name] = 0
    
    total_selected = len(selected_for_test)
    print(f"\n   Gesamt für Test ausgewählt: {total_selected} Bilder")
    
    print(f"\n3. Verschiebe {total_selected} Dateien...")
    
    moved_count = 0
    failed_count = 0
    
    # Statistiken pro Klasse
    class_stats = {}
    
    for img_file in tqdm(selected_for_test, desc="Verschiebe Dateien"):
        try:
            # Klassenname aus Pfad extrahieren
            class_name = img_file.parent.name
            if class_name not in class_stats:
                class_stats[class_name] = 0
            
            # Zieldatei-Pfad (flache Struktur)
            target_file = target_path / img_file.name
            
            # Falls Datei bereits existiert, einen eindeutigen Namen erstellen
            counter = 1
            original_target = target_file
            while target_file.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_file = target_path / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Datei verschieben
            shutil.move(str(img_file), str(target_file))
            moved_count += 1
            class_stats[class_name] += 1
            
        except Exception as e:
            print(f"   Fehler bei {img_file.name}: {e}")
            failed_count += 1
    
    # Ergebnisse anzeigen
    print(f"\n" + "=" * 60)
    print("VERSCHIEBUNG ABGESCHLOSSEN")
    print("=" * 60)
    print(f"✅ Erfolgreich verschoben: {moved_count} Dateien")
    print(f"❌ Fehlgeschlagen: {failed_count} Dateien")
    
    print(f"\nVERTEILUNG IM TEST-SET:")
    total_moved = sum(class_stats.values())
    for class_name, count in sorted(class_stats.items()):
        percentage = (count / total_moved) * 100 if total_moved > 0 else 0
        print(f"   {class_name}: {count} Bilder ({percentage:.1f}%)")
    
    print(f"\nZielverzeichnis: {target_path.absolute()}")
    print(f"Alle Test-Bilder sind jetzt in einer flachen Struktur gespeichert.")
    
    return moved_count, failed_count, class_stats

def verify_move_operation(source_dir="Webscraper/data/resized_mushrooms", 
                         target_dir="Webscraper/data/test_mushrooms"):
    """
    Überprüft das Ergebnis der Verschiebeoperation
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"\nVERIFIKATION:")
    print("=" * 40)
    
    # Zähle verbleibende Bilder in Quellordnern
    remaining_images = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    print("Verbleibende Bilder im Training:")
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_images = []
        for ext in image_extensions:
            class_images.extend(list(class_dir.glob(f"*{ext}")))
            class_images.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        remaining_images += len(class_images)
        print(f"   {class_dir.name}: {len(class_images)} Bilder")
    
    # Zähle Bilder im Test-Ordner
    test_images = []
    if target_path.exists():
        for ext in image_extensions:
            test_images.extend(list(target_path.glob(f"*{ext}")))
            test_images.extend(list(target_path.glob(f"*{ext.upper()}")))
    
    print(f"\nTest-Bilder: {len(test_images)}")
    print(f"Training-Bilder: {remaining_images}")
    print(f"Gesamt: {len(test_images) + remaining_images}")
    
    if len(test_images) > 0:
        test_ratio = len(test_images) / (len(test_images) + remaining_images)
        print(f"Test-Anteil: {test_ratio*100:.1f}%")

def preview_move_operation(source_dir="Webscraper/data/resized_mushrooms", 
                          test_percentage=0.2):
    """
    Zeigt eine Vorschau der geplanten Verschiebeoperation
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Fehler: Quellverzeichnis {source_path} existiert nicht!")
        return
    
    print("VORSCHAU - Geplante Test-Set Erstellung")
    print("=" * 50)
    print(f"Quellverzeichnis: {source_path.absolute()}")
    print(f"Test-Prozentsatz: {test_percentage*100:.0f}% PRO KLASSE")
    
    # Analysiere verfügbare Daten
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    total_images = 0
    total_test_images = 0
    class_counts = {}
    
    print(f"\nKLASSENWEISE AUFTEILUNG:")
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_images = []
        for ext in image_extensions:
            class_images.extend(list(class_dir.glob(f"*{ext}")))
            class_images.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        class_count = len(class_images)
        test_count_in_class = int(class_count * test_percentage)
        remaining_in_class = class_count - test_count_in_class
        
        class_counts[class_dir.name] = class_count
        total_images += class_count
        total_test_images += test_count_in_class
        
        print(f"   {class_dir.name}:")
        print(f"      Gesamt: {class_count} Bilder")
        print(f"      → Test: {test_count_in_class} Bilder ({test_percentage*100:.0f}%)")
        print(f"      → Training: {remaining_in_class} Bilder ({(1-test_percentage)*100:.0f}%)")
    
    train_count = total_images - total_test_images
    actual_test_percentage = (total_test_images / total_images * 100) if total_images > 0 else 0
    
    print(f"\nGESAMT-STATISTIK:")
    print(f"   Test-Set: {total_test_images} Bilder ({actual_test_percentage:.1f}%)")
    print(f"   Training-Set: {train_count} Bilder ({100-actual_test_percentage:.1f}%)")
    print(f"   Gesamt: {total_images} Bilder")
    
    print(f"\nHinweis: Alle Test-Bilder werden in einer flachen Struktur")
    print(f"in 'Webscraper/data/test_mushrooms' gespeichert.")
    print(f"Jede Klasse wird separat mit {test_percentage*100:.0f}% aufgeteilt.")

if __name__ == "__main__":
    print("🔄 RANDOM TEST-SET CREATOR 🔄")
    print("=" * 50)
    
    # Vorschau anzeigen
    preview_move_operation()
    
    # Benutzer fragen
    print("\n" + "=" * 50)
    choice = input("Möchten Sie mit der Verschiebung fortfahren? (j/n): ").lower().strip()
    
    if choice in ['j', 'ja', 'y', 'yes']:
        print("\nStarte Verschiebeoperation...")
        
        # Verschiebung durchführen
        moved, failed, stats = move_random_test_images()
        
        if moved > 0:
            print("\n" + "="*50)
            print("✅ OPERATION ERFOLGREICH ABGESCHLOSSEN!")
            
            # Verifikation
            verify_move_operation()
        else:
            print("\n❌ Keine Dateien verschoben!")
    else:
        print("Operation abgebrochen.")
