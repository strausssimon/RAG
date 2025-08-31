"""
====================================================
Programmname : Create Augmentation Examples
Beschreibung : Erstellt Beispiele für Bildaugmentierungen.

====================================================
"""
import cv2
import numpy as np
from pathlib import Path
import os

class AugmentationExampleCreator:
    def __init__(self, source_dir="Webscraper/data/resized_mushrooms", 
                 example_dir="Webscraper/data/augmented_mushrooms/example"):
        self.source_dir = Path(source_dir)
        self.example_dir = Path(example_dir)
        self.example_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Zoomen zwischen 80% und 120%, behält 128x128 bei"""
        height, width = image.shape[:2]
        
        if zoom_factor > 1.0:
            # Zoom in: Bild vergrößern und dann croppen
            new_height = int(height * zoom_factor)
            new_width = int(width * zoom_factor)
            
            # Bild vergrößern
            enlarged = cv2.resize(image, (new_width, new_height))
            
            # Zentralen Bereich croppen (zurück auf 128x128)
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            zoomed = enlarged[start_y:start_y + height, start_x:start_x + width]
            
        else:
            # Zoom out: Bild verkleinern und mit Rand auffüllen
            new_height = int(height * zoom_factor)
            new_width = int(width * zoom_factor)
            
            # Bild verkleinern
            reduced = cv2.resize(image, (new_width, new_height))
            
            # Schwarzen Hintergrund erstellen (128x128)
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
    
    def create_comparison_image(self, original, augmented, title):
        """Erstellt ein Vergleichsbild (vorher/nachher)"""
        # Titel-Text hinzufügen
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Text-Dimensionen berechnen
        (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)
        
        # Erstelle eine größere Leinwand für Titel
        canvas_height = original.shape[0] * 2 + 80  # Platz für Titel und Bilder
        canvas_width = original.shape[1] + 20  # Etwas Rand
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Titel hinzufügen
        title_x = (canvas_width - text_width) // 2
        cv2.putText(canvas, title, (title_x, 25), font, font_scale, color, thickness)
        
        # "Original" und "Augmented" Labels
        cv2.putText(canvas, "Original", (10, 55), font, 0.4, color, 1)
        cv2.putText(canvas, "Augmented", (10, original.shape[0] + 75), font, 0.4, color, 1)
        
        # Bilder einfügen
        y_original = 60
        y_augmented = y_original + original.shape[0] + 20
        
        canvas[y_original:y_original + original.shape[0], 10:10 + original.shape[1]] = original
        canvas[y_augmented:y_augmented + augmented.shape[0], 10:10 + augmented.shape[1]] = augmented
        
        return canvas
    
    def create_all_examples(self):
        """Erstellt Beispiele für alle Augmentierungstechniken"""
        if not self.source_dir.exists():
            print(f"Fehler: Quell-Ordner {self.source_dir} existiert nicht!")
            return
        
        # Erstes verfügbares Bild finden
        example_image_path = None
        for class_dir in self.source_dir.iterdir():
            if class_dir.is_dir():
                # Alle Bildformate finden
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
                for ext in image_extensions:
                    files = list(class_dir.glob(ext)) + list(class_dir.glob(ext.upper()))
                    if files:
                        example_image_path = files[0]
                        break
                if example_image_path:
                    break
        
        if not example_image_path:
            print("Kein Beispielbild gefunden!")
            return
        
        print(f"Verwende Beispielbild: {example_image_path}")
        print(f"Speichere Beispiele in: {self.example_dir.absolute()}")
        
        # Original Bild laden
        original_image = cv2.imread(str(example_image_path))
        if original_image is None:
            print(f"Fehler beim Laden von {example_image_path}")
            return
        
        # Original speichern
        cv2.imwrite(str(self.example_dir / "00_original.jpg"), original_image)
        
        # Alle Augmentierungstechniken mit Beispielen
        augmentation_examples = [
            ("01_horizontal_flip", "Horizontales Spiegeln", 
             lambda img: self.horizontal_flip(img)),
             
            ("02_rotate_15_pos", "Rotation +15 Grad", 
             lambda img: self.rotate_image(img, 15)),
             
            ("03_rotate_15_neg", "Rotation -15 Grad", 
             lambda img: self.rotate_image(img, -15)),
             
            ("04_rotate_30_pos", "Rotation +30 Grad", 
             lambda img: self.rotate_image(img, 30)),
             
            ("05_zoom_80", "Zoom 80% (Zoom Out)", 
             lambda img: self.zoom_image(img, 0.8)),
             
            ("06_zoom_90", "Zoom 90%", 
             lambda img: self.zoom_image(img, 0.9)),
             
            ("07_zoom_110", "Zoom 110%", 
             lambda img: self.zoom_image(img, 1.1)),
             
            ("08_zoom_120", "Zoom 120% (Zoom In)", 
             lambda img: self.zoom_image(img, 1.2)),
             
            ("09_translate", "Verschiebung (15x, 15y)", 
             lambda img: self.translate_image(img, 15, 15)),
             
            ("10_brightness_plus", "Helligkeit +10%", 
             lambda img: self.adjust_brightness(img, 1.1)),
             
            ("11_brightness_minus", "Helligkeit -10%", 
             lambda img: self.adjust_brightness(img, 0.9)),
             
            ("12_chroma_plus", "Sättigung +10%", 
             lambda img: self.adjust_chroma(img, 1.1)),
             
            ("13_chroma_minus", "Sättigung -10%", 
             lambda img: self.adjust_chroma(img, 0.9)),
             
            ("14_contrast_plus", "Kontrast +10%", 
             lambda img: self.adjust_contrast(img, 1.1)),
             
            ("15_contrast_minus", "Kontrast -10%", 
             lambda img: self.adjust_contrast(img, 0.9)),
             
            ("16_sharpness_plus", "Schärfe +10%", 
             lambda img: self.adjust_sharpness(img, 1.1)),
             
            ("17_sharpness_minus", "Schärfe -10%", 
             lambda img: self.adjust_sharpness(img, 0.9)),
             
            ("18_noise", "Rauschen hinzufügen", 
             lambda img: self.add_noise(img, 20)),
        ]
        
        print(f"\nErstelle {len(augmentation_examples)} Augmentierungsbeispiele...")
        
        for filename, description, aug_func in augmentation_examples:
            try:
                # Augmentierung anwenden
                augmented = aug_func(original_image)
                
                # Einzelne Bilder speichern
                cv2.imwrite(str(self.example_dir / f"{filename}_einzeln.jpg"), augmented)
                
                # Vergleichsbild erstellen (vorher/nachher)
                comparison = self.create_comparison_image(original_image, augmented, description)
                cv2.imwrite(str(self.example_dir / f"{filename}_vergleich.jpg"), comparison)
                
                print(f"✓ {description} - gespeichert als {filename}")
                
            except Exception as e:
                print(f"✗ Fehler bei {description}: {e}")
        
        # Zusammenfassung erstellen
        self.create_summary_info()
        
        print(f"\nAlle Beispiele erfolgreich erstellt!")
        print(f"Gespeichert in: {self.example_dir.absolute()}")
        print(f"Für jede Technik gibt es zwei Dateien:")
        print(f"   - *_einzeln.jpg (nur das augmentierte Bild)")
        print(f"   - *_vergleich.jpg (vorher/nachher Vergleich)")
    
    def create_summary_info(self):
        """Erstellt eine Textdatei mit Zusammenfassung aller Techniken"""
        summary_path = self.example_dir / "augmentation_techniques_summary.txt"
        
        summary_text = """AUGMENTIERUNGSTECHNIKEN ÜBERSICHT
=====================================

Dieses Verzeichnis enthält Beispiele für alle verwendeten Augmentierungstechniken:

1. GEOMETRISCHE TRANSFORMATIONEN:
   - Horizontales Spiegeln: Spiegelt das Bild horizontal
   - Rotation: Dreht das Bild um ±15° und ±30°
   - Zoom: Vergrößert (110%, 120%) oder verkleinert (80%, 90%) das Bild
   - Translation: Verschiebt das Bild um 15 Pixel in X- und Y-Richtung

2. PHOTOMETRISCHE ANPASSUNGEN:
   - Helligkeit: Erhöht (+10%) oder reduziert (-10%) die Bildhelligkeit
   - Sättigung/Chroma: Verstärkt (+10%) oder reduziert (-10%) die Farbsättigung
   - Kontrast: Erhöht (+10%) oder reduziert (-10%) den Bildkontrast
   - Schärfe: Schärft (+10%) oder weichzeichnet (-10%) das Bild

3. RAUSCHEN:
   - Gaußsches Rauschen: Fügt zufälliges Rauschen hinzu für Robustheit

DATEIFORMATE:
- *_einzeln.jpg: Nur das augmentierte Bild (128x128 Pixel)
- *_vergleich.jpg: Vorher/Nachher Vergleich

ZWECK:
Diese Augmentierungen verbessern die Generalisierungsfähigkeit des CNN-Modells,
indem sie die Trainingsdaten diversifizieren und das Modell robuster gegen
Variationen in echten Bildern machen.

Alle Augmentierungen behalten die Ausgabegröße von 128x128 Pixeln bei.
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"Zusammenfassung gespeichert: {summary_path.name}")

if __name__ == "__main__":
    print("AUGMENTATION EXAMPLE CREATOR")
    print("=" * 50)
    print("Erstellt Beispiele für alle Augmentierungstechniken")
    print("Speicherort: Webscraper/data/augmented_mushrooms/example")
    print()
    
    creator = AugmentationExampleCreator()
    creator.create_all_examples()
    
    print("\n" + "=" * 50)
    print("Fertig! Sie können nun die Beispiele anschauen.")
