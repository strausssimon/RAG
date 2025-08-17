import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

def resize_images_in_directory(source_path, target_path, target_size=(128, 128)):
    """
    Resized alle Bilder in einem Verzeichnis auf die gewünschte Größe
    und speichert sie in einem separaten Zielverzeichnis mit gleicher Struktur
    
    Args:
        source_path (str): Pfad zum Quellverzeichnis
        target_path (str): Pfad zum Zielverzeichnis
        target_size (tuple): Zielgröße (width, height)
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    if not source_path.exists():
        print(f"Fehler: Quellpfad {source_path} existiert nicht!")
        return
    
    # Zielverzeichnis erstellen falls es nicht existiert
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Quellverzeichnis: {source_path}")
    print(f"Zielverzeichnis: {target_path}")
    
    # Unterstützte Bildformate
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Alle Bildateien finden (rekursiv durch alle Unterordner)
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(source_path.rglob(f"*{ext}")))
        all_images.extend(list(source_path.rglob(f"*{ext.upper()}")))
    
    print(f"Gefunden: {len(all_images)} Bilder zum Resizen")
    print(f"Zielgröße: {target_size[0]}x{target_size[1]} Pixel")
    
    if len(all_images) == 0:
        print("Keine Bilder gefunden!")
        return
    
    # Statistiken
    successful_resizes = 0
    failed_resizes = 0
    skipped_files = 0
    
    # Durch alle Bilder iterieren
    for img_path in tqdm(all_images, desc="Resizing images"):
        try:
            # Bild laden
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Warnung: Kann {img_path.name} nicht laden")
                failed_resizes += 1
                continue
            
            # Relative Pfadstruktur vom Quellverzeichnis beibehalten
            relative_path = img_path.relative_to(source_path)
            target_img_path = target_path / relative_path
            
            # ⭐ WICHTIG: Zielverzeichnis für das Bild erstellen (alle Unterordner werden automatisch angelegt!)
            target_img_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"📁 Erstelle Verzeichnis: {target_img_path.parent}")  # Debug-Info
            
            # Vorhandene Dateien werden überschrieben
            if target_img_path.exists():
                print(f"🔄 Überschreibe vorhandene Datei: {target_img_path.name}")
                # Datei wird überschrieben - kein Skip
            
            # Bild resizen
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Resized Bild in Zielverzeichnis speichern
            success = cv2.imwrite(str(target_img_path), resized_img)
            
            if success:
                successful_resizes += 1
            else:
                print(f"Fehler beim Speichern von {target_img_path.name}")
                failed_resizes += 1
                
        except Exception as e:
            print(f"Fehler bei {img_path.name}: {str(e)}")
            failed_resizes += 1
    
    # Abschlussbericht
    print(f"\n=== RESIZE ABGESCHLOSSEN ===")
    print(f"✅ Erfolgreich resized: {successful_resizes}")
    print(f"⏭️ Übersprungen (bereits existiert): {skipped_files}")
    print(f"❌ Fehlgeschlagen: {failed_resizes}")
    print(f"📊 Gesamt verarbeitet: {len(all_images)}")
    print(f"📁 Gespeichert in: {target_path}")

def verify_resize(check_path, expected_size=(128, 128)):
    """
    Überprüft ob alle Bilder die erwartete Größe haben
    """
    check_path = Path(check_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(check_path.rglob(f"*{ext}")))
        all_images.extend(list(check_path.rglob(f"*{ext.upper()}")))
    
    print(f"\n=== VERIFIKATION ===")
    print(f"Überprüfe {len(all_images)} Bilder in {check_path}...")
    
    correct_size = 0
    wrong_size = 0
    unreadable = 0
    
    for img_path in tqdm(all_images, desc="Verifying sizes"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                unreadable += 1
                continue
                
            height, width = img.shape[:2]
            if width == expected_size[0] and height == expected_size[1]:
                correct_size += 1
            else:
                wrong_size += 1
                print(f"Falsche Größe: {img_path.name} - {width}x{height}")
                
        except Exception as e:
            unreadable += 1
            print(f"Fehler bei {img_path.name}: {str(e)}")
    
    print(f"✅ Korrekte Größe ({expected_size[0]}x{expected_size[1]}): {correct_size}")
    print(f"❌ Falsche Größe: {wrong_size}")
    print(f"📷 Nicht lesbar: {unreadable}")

def count_images_by_class(directory_path):
    """
    Zählt Bilder pro Klasse in der Verzeichnisstruktur
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Verzeichnis {directory_path} existiert nicht!")
        return
    
    print(f"\n=== BILDANZAHL PRO KLASSE ===")
    
    # Alle Unterverzeichnisse durchgehen
    for class_dir in directory_path.iterdir():
        if class_dir.is_dir():
            # Bilder in diesem Klassen-Verzeichnis zählen
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                image_files.extend(list(class_dir.glob(f"*{ext}")))
                image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            print(f"📁 {class_dir.name}: {len(image_files)} Bilder")

if __name__ == "__main__":
    # Pfade definieren (korrekte relative Pfade von src/helpers aus)
    source_directory = "Webscraper/data/images_mushrooms"
    target_directory = "Webscraper/data/resized_mushrooms"
    resize_target = (200, 200)
    
    print("🍄 MUSHROOM IMAGE RESIZER 🍄")
    print("=" * 50)
    
    # Originalbilder zählen
    print("Originalbilder:")
    count_images_by_class(source_directory)
    
    # Bilder resizen und in separates Verzeichnis speichern
    resize_images_in_directory(source_directory, target_directory, resize_target)
    
    # Verifikation der resized Bilder
    verify_resize(target_directory, resize_target)
    
    # Resized Bilder zählen
    print("\nResized Bilder:")
    count_images_by_class(target_directory)
    
    print("\n" + "="*50)
    print("✅ Resize-Vorgang abgeschlossen!")
    print(f"📂 Originale bleiben unverändert in: {source_directory}")
    print(f"📂 Resized Bilder (200x200) gespeichert in: {target_directory}")
    print("🔄 Dateistruktur wurde beibehalten")
