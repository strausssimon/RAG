import cv2
from pathlib import Path
from tqdm import tqdm
import os

def resize_images_in_place(directory, target_size=(200, 200)):
    """
    Resized alle Bilder im angegebenen Verzeichnis (rekursiv) auf die gewünschte Größe (in-place).
    Die Bilder werden überschrieben und bleiben im selben Ordner.
    
    Args:
        directory (str): Pfad zum Verzeichnis mit den Bildern
        target_size (tuple): Zielgröße (width, height)
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Fehler: Verzeichnis {directory} existiert nicht!")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(directory.rglob(f"*{ext}")))
        all_images.extend(list(directory.rglob(f"*{ext.upper()}")))

    print(f"Gefunden: {len(all_images)} Bilder zum Resizen in {directory}")
    print(f"Zielgröße: {target_size[0]}x{target_size[1]} Pixel")

    if len(all_images) == 0:
        print("Keine Bilder gefunden!")
        return

    successful_resizes = 0
    failed_resizes = 0

    for img_path in tqdm(all_images, desc="Resizing images in-place"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warnung: Kann {img_path.name} nicht laden")
                failed_resizes += 1
                continue
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            success = cv2.imwrite(str(img_path), resized_img)
            if success:
                successful_resizes += 1
            else:
                print(f"Fehler beim Speichern von {img_path.name}")
                failed_resizes += 1
        except Exception as e:
            print(f"Fehler bei {img_path.name}: {str(e)}")
            failed_resizes += 1

    print(f"\n=== RESIZE ABGESCHLOSSEN ===")
    print(f"✅ Erfolgreich resized: {successful_resizes}")
    print(f"❌ Fehlgeschlagen: {failed_resizes}")
    print(f"📊 Gesamt verarbeitet: {len(all_images)}")
    print(f"📁 Verzeichnis: {directory}")

if __name__ == "__main__":
    # Passe den Pfad ggf. an
    test_dir = Path(__file__).parent.parent.parent / "data" / "test_mushrooms_randomized"
    resize_images_in_place(test_dir, target_size=(200, 200))
