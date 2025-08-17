import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import random

class ImageAugmenter:
    def __init__(self, source_dir="data/resized_mushrooms", 
                 output_dir="data/augmented_mushrooms/resized"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def horizontal_flip(self, image):
        """Horizontales Spiegeln"""
        return cv2.flip(image, 1)
        
    def rotate_image(self, image, angle):
        """Rotiert das Bild um einen bestimmten Winkel"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
    def zoom_image(self, image, zoom_factor=1.2):
        """Zoomen zwischen 80% und 120%, behält 200x200 bei"""
        height, width = image.shape[:2]
        
        if zoom_factor > 1.0:
            # Zoom in (120%): Bild vergrößern und dann croppen
            new_height = int(height * zoom_factor)
            new_width = int(width * zoom_factor)
            
            # Bild vergrößern
            enlarged = cv2.resize(image, (new_width, new_height))
            
            # Zentralen Bereich croppen (zurück auf 200x200)
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            zoomed = enlarged[start_y:start_y + height, start_x:start_x + width]
            
        else:
            # Zoom out (80%): Bild verkleinern und mit Rand auffüllen
            new_height = int(height * zoom_factor)
            new_width = int(width * zoom_factor)
            
            # Bild verkleinern
            reduced = cv2.resize(image, (new_width, new_height))
            
            # Schwarzen Hintergrund erstellen (200x200)
            zoomed = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Verkleinertes Bild zentriert einfügen
            start_y = (height - new_height) // 2
            start_x = (width - new_width) // 2
            zoomed[start_y:start_y + new_height, start_x:start_x + new_width] = reduced
            
        return zoomed
    
    def translate_image(self, image, shift_x=15, shift_y=15):
        """Verschieben des Bildes"""
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated = cv2.warpAffine(image, translation_matrix, (width, height))
        return translated
    
    def flip_image(self, image, flip_code):
        """Spiegelt das Bild (0=vertikal, 1=horizontal, -1=beide)"""
        return cv2.flip(image, flip_code)
    
    def adjust_brightness(self, image, factor):
        """Passt die Helligkeit an"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_chroma(self, image, factor):
        """Passt die Chroma (Farbsättigung) an"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor  # Sättigung anpassen
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """Passt den Kontrast an"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def adjust_sharpness(self, image, factor):
        """Passt die Schärfe an"""
        # Schärfe-Kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        # Original und geschärftes Bild kombinieren
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Gewichtete Kombination basierend auf dem Faktor
        if factor > 1.0:
            # Schärfen
            blend_factor = min((factor - 1.0), 1.0)
            result = cv2.addWeighted(image, 1 - blend_factor, sharpened, blend_factor, 0)
        else:
            # Weichzeichnen (Schärfe reduzieren)
            blur_strength = int((1.0 - factor) * 5) + 1
            if blur_strength % 2 == 0:
                blur_strength += 1  # Kernel-Größe muss ungerade sein
            blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
            result = blurred
        
        return result
    
    def add_noise(self, image, noise_level=25):
        """Fügt Rauschen hinzu"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def blur_image(self, image, kernel_size=5):
        """Verwischung des Bildes"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def crop_and_resize(self, image, crop_factor=0.8):
        """Beschneidet das Bild und skaliert es zurück"""
        height, width = image.shape[:2]
        new_height = int(height * crop_factor)
        new_width = int(width * crop_factor)
        
        start_y = (height - new_height) // 2
        start_x = (width - new_width) // 2
        
        cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]
        resized = cv2.resize(cropped, (width, height))
        return resized
    
    def augment_single_image(self, image_path, class_name, num_augmentations=6):
        """Erstellt mehrere augmentierte Versionen eines Bildes"""
        # Zielordner für diese Klasse erstellen
        class_output_dir = self.output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Original Bild laden
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f"Fehler beim Laden von {image_path}")
            return 0
        
        augmented_count = 0
        
        # Original Bild speichern
        original_name = f"{image_path.stem}_original.jpg"
        cv2.imwrite(str(class_output_dir / original_name), original_image)
        augmented_count += 1
        
        # Optimierte Augmentationen für bessere Generalisierung
        augmentations = [
            ("flip_h", lambda img: self.horizontal_flip(img)),
            ("rot15", lambda img: self.rotate_image(img, 15)),
            ("rot_15", lambda img: self.rotate_image(img, -15)),
            ("rot30", lambda img: self.rotate_image(img, 30)),
            # Zoom-Variationen: 80%, 90%, 110%, 120%
            ("zoom_80", lambda img: self.zoom_image(img, 0.8)),
            ("zoom_90", lambda img: self.zoom_image(img, 0.9)),
            ("zoom_110", lambda img: self.zoom_image(img, 1.1)),
            ("zoom_120", lambda img: self.zoom_image(img, 1.2)),
            ("translate", lambda img: self.translate_image(img, 15, 10)),
            # Helligkeit: +10% (1.1x) und -10% (0.9x)
            ("bright_plus", lambda img: self.adjust_brightness(img, 1.1)),
            ("bright_minus", lambda img: self.adjust_brightness(img, 0.9)),
            # Chroma/Sättigung: +10% (1.1x) und -10% (0.9x)
            ("chroma_plus", lambda img: self.adjust_chroma(img, 1.1)),
            ("chroma_minus", lambda img: self.adjust_chroma(img, 0.9)),
            # Kontrast: +10% (1.1x) und -10% (0.9x)
            ("contrast_plus", lambda img: self.adjust_contrast(img, 1.1)),
            ("contrast_minus", lambda img: self.adjust_contrast(img, 0.9)),
            # Schärfe: +10% (1.1x) und -10% (0.9x)
            ("sharp_plus", lambda img: self.adjust_sharpness(img, 1.1)),
            ("sharp_minus", lambda img: self.adjust_sharpness(img, 0.9)),
            ("noise", lambda img: self.add_noise(img, 20)),
        ]
        
        # Zufällige Auswahl von Augmentationen
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        for aug_name, aug_func in selected_augmentations:
            try:
                augmented_image = aug_func(original_image)
                aug_filename = f"{image_path.stem}_{aug_name}.jpg"
                cv2.imwrite(str(class_output_dir / aug_filename), augmented_image)
                augmented_count += 1
            except Exception as e:
                print(f"Fehler bei Augmentation {aug_name} für {image_path.name}: {e}")
        
        return augmented_count

    def augment_mushroom_dataset(self, augmentations_per_image=6):
        """Augmentiert den gesamten resized Pilz-Datensatz"""
        if not self.source_dir.exists():
            print(f"Fehler: Quell-Ordner {self.source_dir} existiert nicht!")
            return
        
        print(f"Starte Datenaugmentation von resized Bildern...")
        print(f"Quell-Ordner: {self.source_dir.absolute()}")
        print(f"Ziel-Ordner: {self.output_dir.absolute()}")
        print("=" * 60)
        
        total_original = 0
        total_augmented = 0
        
        # Alle Unterordner durchgehen (Pilzklassen)
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            print(f"\nVerarbeite Klasse: {class_name}...")
            
            # Alle Bildformate finden
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(ext)))
                image_files.extend(list(class_dir.glob(ext.upper())))
            
            if not image_files:
                print(f"   Keine Bilder in {class_name} gefunden")
                continue
            
            print(f"   Gefunden: {len(image_files)} Originalbilder")
            total_original += len(image_files)
            
            # Jedes Bild augmentieren
            class_augmented = 0
            for img_file in tqdm(image_files, desc=f"Augmentiere {class_name}"):
                augmented_count = self.augment_single_image(
                    img_file, class_name, augmentations_per_image
                )
                class_augmented += augmented_count
            
            print(f"   Erstellt: {class_augmented} augmentierte Bilder (inkl. Original)")
            total_augmented += class_augmented
        
        print("\n" + "=" * 60)
        print("Augmentation abgeschlossen!")
        print(f"Originale Bilder: {total_original}")
        print(f"Augmentierte Bilder (inkl. Original): {total_augmented}")
        print(f"Vervielfachung: {total_augmented/total_original:.1f}x")
        print(f"Alle Dateien in: {self.output_dir.absolute()}")

def preview_augmentation():
    """Zeigt eine Vorschau der geplanten Augmentation für resized Bilder"""
    source_path = Path("data/resized_mushrooms")
    
    print("VORSCHAU - Geplante Datenaugmentation für resized Bilder")
    print("=" * 60)
    print(f"Quellverzeichnis: {source_path}")
    
    if not source_path.exists():
        print(f"FEHLER: Quellverzeichnis {source_path} existiert nicht!")
        return
    
    total_files = 0
    classes_found = []
    
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            # Alle Bildformate zählen
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(ext)))
                image_files.extend(list(class_dir.glob(ext.upper())))
            
            class_count = len(image_files)
            print(f"{class_dir.name}: {class_count} Bilder")
            total_files += class_count
            classes_found.append(class_dir.name)
    
    if not classes_found:
        print("Keine Pilzklassen gefunden!")
        return
    
    augmentations_per_image = 7  # Original + 6 Augmentationen
    estimated_total = total_files * augmentations_per_image
    
    print(f"\nGefundene Klassen: {', '.join(classes_found)}")
    print(f"\nGeschätzte Ergebnisse:")
    print(f"Original Bilder: {total_files}")
    print(f"Augmentationen pro Bild: {augmentations_per_image} (inkl. Original)")
    print(f"Geschätzte Gesamtanzahl: {estimated_total} Bilder")
    print(f"\nAngewendete Augmentationen für bessere Generalisierung:")
    print("- Horizontales Spiegeln")
    print("- Rotieren (±15°, ±30°)")
    print("- Zoomen (80%, 90%, 110%, 120% - 200x200 bleibt erhalten)")
    print("- Verschieben")
    print("- Helligkeit: +10% (1.1x) und -10% (0.9x)")
    print("- Chroma/Sättigung: +10% (1.1x) und -10% (0.9x)")
    print("- Kontrast: +10% (1.1x) und -10% (0.9x)")
    print("- Schärfe: +10% (1.1x) und -10% (0.9x)")
    print("- Rauschen hinzufügen")

if __name__ == "__main__":
    print("🍄 MUSHROOM DATASET AUGMENTER - RESIZED IMAGES 🍄")
    print("=" * 50)
    
    # Vorschau anzeigen
    preview_augmentation()
    
    # Benutzer fragen
    print("\n" + "=" * 50)
    choice = input("Möchten Sie mit der Augmentation fortfahren? (j/n): ").lower().strip()
    
    if choice in ['j', 'ja', 'y', 'yes']:
        print("\nStarte Augmentation...")
        
        # Anzahl Augmentationen pro Bild fragen
        try:
            num_aug = int(input("Augmentationen pro Bild (empfohlen: 5-8): ") or "6")
        except ValueError:
            num_aug = 6
        
        augmenter = ImageAugmenter()
        augmenter.augment_mushroom_dataset(augmentations_per_image=num_aug)
        
    else:
        print("Augmentation abgebrochen.")
