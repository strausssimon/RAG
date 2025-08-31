"""
====================================================
Programmname : Augmentation
Beschreibung : Augmentierungstechniken für Bilddaten

====================================================
"""
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import random

class ImageAugmenter:
    def __init__(self, output_dir="Webscraper/data/augmented_cropped_mushrooms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def rotate_image(self, image, angle):
        """Rotiert das Bild um einen bestimmten Winkel"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    
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
    
    def adjust_contrast(self, image, factor):
        """Passt den Kontrast an"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
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
    
    def augment_single_image(self, image_path, output_prefix, num_augmentations=5):
        """Erstellt mehrere augmentierte Versionen eines Bildes"""
        # Original Bild laden
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f"Fehler beim Laden von {image_path}")
            return 0
        
        augmented_count = 0
        
        # Original Bild speichern
        original_name = f"{output_prefix}_original_{image_path.stem}.jpg"
        cv2.imwrite(str(self.output_dir / original_name), original_image)
        augmented_count += 1
        
        # Verschiedene Augmentationen anwenden
        augmentations = [
            ("rot15", lambda img: self.rotate_image(img, 15)),
            ("rot30", lambda img: self.rotate_image(img, 30)),
            ("rot_15", lambda img: self.rotate_image(img, -15)),
            ("rot_30", lambda img: self.rotate_image(img, -30)),
            ("flip_h", lambda img: self.flip_image(img, 1)),
            ("flip_v", lambda img: self.flip_image(img, 0)),
            ("bright1.2", lambda img: self.adjust_brightness(img, 1.2)),
            ("bright0.8", lambda img: self.adjust_brightness(img, 0.8)),
            ("contrast1.3", lambda img: self.adjust_contrast(img, 1.3)),
            ("contrast0.7", lambda img: self.adjust_contrast(img, 0.7)),
            ("noise", lambda img: self.add_noise(img, 20)),
            ("blur", lambda img: self.blur_image(img, 3)),
            ("crop", lambda img: self.crop_and_resize(img, 0.85)),
        ]
        
        # Zufällige Auswahl von Augmentationen
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        for aug_name, aug_func in selected_augmentations:
            try:
                augmented_image = aug_func(original_image)
                aug_filename = f"{output_prefix}_{aug_name}_{image_path.stem}.jpg"
                cv2.imwrite(str(self.output_dir / aug_filename), augmented_image)
                augmented_count += 1
            except Exception as e:
                print(f"Fehler bei Augmentation {aug_name} für {image_path.name}: {e}")
        
        # Kombinierte Augmentationen
        try:
            # Rotation + Helligkeit
            combined1 = self.adjust_brightness(self.rotate_image(original_image, 20), 1.1)
            combined1_name = f"{output_prefix}_rot20_bright_{image_path.stem}.jpg"
            cv2.imwrite(str(self.output_dir / combined1_name), combined1)
            augmented_count += 1
            
            # Flip + Kontrast
            combined2 = self.adjust_contrast(self.flip_image(original_image, 1), 1.2)
            combined2_name = f"{output_prefix}_flip_contrast_{image_path.stem}.jpg"
            cv2.imwrite(str(self.output_dir / combined2_name), combined2)
            augmented_count += 1
            
        except Exception as e:
            print(f"Fehler bei kombinierten Augmentationen für {image_path.name}: {e}")
        
        return augmented_count
    
    def augment_mushroom_dataset(self, mushroom_types=["Cropped_Armillaria_mellea", "Cropped_Boletus_edulis"], 
                                 augmentations_per_image=7):
        """Augmentiert den gesamten zugeschnittenen Pilz-Datensatz"""
        base_path = Path("Webscraper/data/cropped_mushrooms")
        
        if not base_path.exists():
            print(f"Fehler: Basis-Ordner {base_path} existiert nicht!")
            return
        
        print(f"Starte Datenaugmentation...")
        print(f"Output-Ordner: {self.output_dir.absolute()}")
        print("=" * 60)
        
        total_original = 0
        total_augmented = 0
        
        for mushroom_type in mushroom_types:
            mushroom_dir = base_path / mushroom_type
            
            if not mushroom_dir.exists():
                print(f"Warnung: Ordner {mushroom_dir} existiert nicht!")
                continue
            
            print(f"\nVerarbeite {mushroom_type}...")
            
            # Alle JPG-Dateien finden
            jpg_files = list(mushroom_dir.glob("*.jpg"))
            
            if not jpg_files:
                print(f"   Keine .jpg Dateien in {mushroom_type} gefunden")
                continue
            
            print(f"   Gefunden: {len(jpg_files)} Originalbilder")
            total_original += len(jpg_files)
            
            # Prefix für Output-Dateien (Label im Namen - ohne "Cropped_")
            if mushroom_type.startswith("Cropped_"):
                prefix = mushroom_type[8:].lower()  # Entfernt "Cropped_" Prefix
            else:
                prefix = mushroom_type.lower()
            
            # Jedes Bild augmentieren
            mushroom_augmented = 0
            for img_file in tqdm(jpg_files, desc=f"Augmentiere {mushroom_type}"):
                augmented_count = self.augment_single_image(
                    img_file, prefix, augmentations_per_image
                )
                mushroom_augmented += augmented_count
            
            print(f"   Erstellt: {mushroom_augmented} augmentierte Bilder")
            total_augmented += mushroom_augmented
        
        print("\n" + "=" * 60)
        print("Augmentation abgeschlossen!")
        print(f"Originale Bilder: {total_original}")
        print(f"Augmentierte Bilder: {total_augmented}")
        print(f"Vervielfachung: {total_augmented/total_original:.1f}x")
        print(f"Gesamt-Datensatz: {total_augmented} Bilder")
        print(f"Alle Dateien in: {self.output_dir.absolute()}")

def preview_augmentation():
    """Zeigt eine Vorschau der geplanten Augmentation für zugeschnittene Bilder"""
    base_path = Path("Webscraper/data/cropped_mushrooms")
    mushroom_types = ["Cropped_Armillaria_mellea", "Cropped_Boletus_edulis"]
    
    print("VORSCHAU - Geplante Datenaugmentation für zugeschnittene Bilder")
    print("=" * 60)
    
    total_files = 0
    for mushroom_type in mushroom_types:
        mushroom_dir = base_path / mushroom_type
        if mushroom_dir.exists():
            jpg_files = list(mushroom_dir.glob("*.jpg"))
            print(f"{mushroom_type}: {len(jpg_files)} Bilder")
            total_files += len(jpg_files)
        else:
            print(f"{mushroom_type}: Ordner nicht gefunden!")
    
    augmentations_per_image = 8  # Original + 7 Augmentationen
    estimated_total = total_files * augmentations_per_image
    
    print(f"\nGeschätzte Ergebnisse:")
    print(f"Original Bilder: {total_files}")
    print(f"Augmentationen pro Bild: {augmentations_per_image}")
    print(f"Geschätzte Gesamtanzahl: {estimated_total} Bilder")

if __name__ == "__main__":
    print("Mushroom Dataset Augmenter - Cropped Images")
    print("=" * 40)
    
    # Vorschau anzeigen
    preview_augmentation()
    
    # Benutzer fragen
    print("\n" + "=" * 50)
    choice = input("Möchten Sie mit der Augmentation fortfahren? (j/n): ").lower().strip()
    
    if choice in ['j', 'ja', 'y', 'yes']:
        print("\nStarte Augmentation...")
        
        # Anzahl Augmentationen pro Bild fragen
        try:
            num_aug = int(input("Augmentationen pro Bild (empfohlen: 5-10): ") or "7")
        except ValueError:
            num_aug = 7
        
        augmenter = ImageAugmenter()
        augmenter.augment_mushroom_dataset(augmentations_per_image=num_aug)
    else:
        print("Augmentation abgebrochen.")
