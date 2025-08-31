"""
====================================================
Programmname : Crop Mushrooms
Beschreibung : Entfernt den Hintergrund von Pilzbildern und schneidet sie auf eine einheitliche Größe zu.

====================================================
"""
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

class MushroomCropper:
    #def __init__(self, source_dir="Webscraper/data/images_mushrooms/Boletus_edulis", 
    def __init__(self, source_dir="data/inaturalist_mushrooms/phallus_impudicus",
                 output_dir="data/cropped_mushrooms/Cropped_phallus_impudicus",
                 #self, source_dir="Webscraper\data\images_mushrooms\Boletus_edulis",
                 #output_dir="Webscraper/data/cropped_mushrooms/Cropped_Boletus_edulis",
                 
                 #output_dir="Webscraper/data/cropped_mushrooms/Cropped_Boletus_edulis",
                 target_size=(224, 224)):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Output-Verzeichnis erstellen
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Mushroom Cropper initialisiert:")
        print(f"  Quelle: {self.source_dir}")
        print(f"  Ziel: {self.output_dir}")
        print(f"  Zielgröße: {target_size}")
    
    def remove_background_grabcut(self, image):
        """
        Entfernt Hintergrund mit GrabCut-Algorithmus (verbessert)
        """
        height, width = image.shape[:2]
        
        # Mehrere Rechtecke testen für bessere Ergebnisse
        rects = [
            # Standard mittig
            (int(width * 0.1), int(height * 0.1), 
             int(width * 0.8), int(height * 0.8)),
            # Etwas kleiner für präzisere Segmentierung
            (int(width * 0.15), int(height * 0.15), 
             int(width * 0.7), int(height * 0.7)),
            # Größer für kleinere Pilze
            (int(width * 0.05), int(height * 0.05), 
             int(width * 0.9), int(height * 0.9))
        ]
        
        best_mask = None
        best_area = 0
        
        for rect in rects:
            try:
                # GrabCut Masken initialisieren
                mask = np.zeros((height, width), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # GrabCut mit mehr Iterationen für bessere Qualität
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)
                
                # Maske erstellen (Vordergrund und wahrscheinlicher Vordergrund)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                # Erweiterte morphologische Operationen
                kernel_small = np.ones((3, 3), np.uint8)
                kernel_medium = np.ones((5, 5), np.uint8)
                
                # Rauschen entfernen
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel_small)
                # Löcher füllen
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel_medium)
                # Konturen glätten
                mask2 = cv2.dilate(mask2, kernel_small, iterations=1)
                mask2 = cv2.erode(mask2, kernel_small, iterations=1)
                
                # Qualität bewerten
                area = np.sum(mask2)
                if area > best_area:
                    best_area = area
                    best_mask = mask2
                    
            except Exception:
                continue
        
        return best_mask
    
    def remove_background_color_segmentation(self, image):
        """
        Entfernt Hintergrund basierend auf Farbsegmentierung (optimiert für zusammenhängende Pilzformen)
        """
        # Bild in HSV konvertieren
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Pilz-spezifische Farbpalette (konservativer für bessere Zusammenhänge)
        masks = []
        
        # Brauntöne (typische Pilzfarben) - erweiterte Bereiche
        lower_brown = np.array([8, 30, 30])
        upper_brown = np.array([30, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        masks.append(brown_mask)
        
        # Beige/Gelbliche Töne - erweitert für Stiele
        lower_beige = np.array([10, 20, 40])
        upper_beige = np.array([40, 200, 255])
        beige_mask = cv2.inRange(hsv, lower_beige, upper_beige)
        masks.append(beige_mask)
        
        # Weiße/helle Töne (Pilzhüte)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        masks.append(white_mask)
        
        # Grau-Töne (typisch für viele Pilze)
        lower_gray = np.array([0, 0, 80])
        upper_gray = np.array([180, 60, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        masks.append(gray_mask)
        
        # Kombinierte Maske - großzügiger für zusammenhängende Formen
        mushroom_mask = np.zeros_like(masks[0])
        for mask in masks:
            mushroom_mask = cv2.bitwise_or(mushroom_mask, mask)
        
        # Aggressive morphologische Operationen für zusammenhängende Formen
        kernel_large = np.ones((15, 15), np.uint8)  # Größerer Kernel
        kernel_medium = np.ones((9, 9), np.uint8)
        kernel_small = np.ones((5, 5), np.uint8)
        
        # 1. Starke Dilatation um Lücken zu schließen
        mushroom_mask = cv2.dilate(mushroom_mask, kernel_large, iterations=3)
        
        # 2. Große Löcher füllen
        mushroom_mask = cv2.morphologyEx(mushroom_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # 3. Kleine Fragmente entfernen
        mushroom_mask = cv2.morphologyEx(mushroom_mask, cv2.MORPH_OPEN, kernel_medium)
        
        # 4. Konturen verfeinern
        mushroom_mask = cv2.erode(mushroom_mask, kernel_small, iterations=2)
        mushroom_mask = cv2.dilate(mushroom_mask, kernel_small, iterations=1)
        
        return mushroom_mask // 255
    
    def remove_background_watershed(self, image):
        """
        Watershed-Segmentierung für zusammenhängende Pilzformen
        """
        # Graustufen für Watershed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 5)
        
        # Threshold für Vordergrund-Marker
        ret, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask from watershed result
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255  # Exclude background (marker 1)
        
        return mask // 255
    
    def find_largest_contour(self, mask):
        """
        Findet die größte zusammenhängende Region (verbessert für Pilzerkennung)
        """
        # Konturen finden
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Konturen nach Qualitätskriterien bewerten
        best_contour = None
        best_score = 0
        
        image_area = mask.shape[0] * mask.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Mindestfläche check
            if area < max(1000, image_area * 0.005):  # Mindestens 0.5% der Bildfläche
                continue
            
            # Kompaktheit berechnen (Kreisähnlichkeit)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect Ratio der Bounding Box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Score basierend auf: Größe, Kompaktheit, vernünftiges Aspect Ratio
            # Pilze sollten nicht zu lang/schmal sein
            aspect_penalty = 1.0
            if aspect_ratio > 3 or aspect_ratio < 0.3:  # Zu schmal oder zu breit
                aspect_penalty = 0.5
            
            score = area * compactness * aspect_penalty
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def create_bounding_box_with_margin(self, contour, image_shape, margin=0.1):
        """
        Erstellt Bounding Box mit Rand um den Pilz
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        # Margin hinzufügen
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image_shape[1] - x, w + 2 * margin_x)
        h = min(image_shape[0] - y, h + 2 * margin_y)
        
        return x, y, w, h
    
    def crop_mushroom(self, image_path):
        """
        Schneidet Pilz basierend auf Kontrastlinien aus und macht Hintergrund schwarz
        """
        try:
            # Bild laden
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                return None
            
            original_height, original_width = original_image.shape[:2]
            
            # Kontrastbasierte Objekterkennung
            detected_contour, edge_map = self.detect_edges_and_contours(original_image)
            
            if detected_contour is None:
                # Fallback: Versuche Farbsegmentierung als Backup
                color_mask = self.remove_background_color_segmentation(original_image)
                if color_mask is not None:
                    detected_contour = self.find_largest_contour(color_mask)
                
                if detected_contour is None:
                    return None
            
            # Bounding Box um erkanntes Objekt
            x, y, w, h = cv2.boundingRect(detected_contour)
            
            # Sicherheitsmargin um das Objekt
            margin_x = max(5, int(w * 0.05))  # 5% Margin, mindestens 5 Pixel
            margin_y = max(5, int(h * 0.05))
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(original_width - x, w + 2 * margin_x)
            h = min(original_height - y, h + 2 * margin_y)
            
            # Quadratisches Cropping für CNN
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Quadratische Region zentrieren
            x_square = max(0, center_x - size // 2)
            y_square = max(0, center_y - size // 2)
            
            # Grenzen prüfen
            if x_square + size > original_width:
                x_square = original_width - size
            if y_square + size > original_height:
                y_square = original_height - size
                
            if x_square < 0 or y_square < 0 or size <= 0:
                # Fallback: Rechteckiges Cropping
                cropped_region = original_image[y:y+h, x:x+w]
                cropped_contour = detected_contour.copy()
                # Kontur an neue Position anpassen
                cropped_contour[:, :, 0] -= x
                cropped_contour[:, :, 1] -= y
            else:
                # Quadratisches Cropping
                cropped_region = original_image[y_square:y_square+size, x_square:x_square+size]
                cropped_contour = detected_contour.copy()
                # Kontur an neue Position anpassen
                cropped_contour[:, :, 0] -= x_square
                cropped_contour[:, :, 1] -= y_square
            
            # Maske für das Objekt erstellen
            mask = np.zeros(cropped_region.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [cropped_contour], 255)
            
            # Hintergrund schwarz machen
            result = cropped_region.copy()
            result[mask == 0] = [0, 0, 0]  # Schwarzer Hintergrund
            
            # Auf Zielgröße skalieren
            if result.size == 0:
                return None
                
            final_result = cv2.resize(result, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return final_result
            
        except Exception as e:
            print(f"Fehler beim Cropping von {image_path}: {e}")
            return None
    
    def process_all_images(self):
        """
        Verarbeitet alle Bilder im Quellordner
        """
        if not self.source_dir.exists():
            print(f"Fehler: Quellordner {self.source_dir} existiert nicht!")
            return
        
        print(f"Suche Bilder in: {self.source_dir.absolute()}")
        
        # Mehrere Dateierweiterungen suchen
        image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        image_files = []
        
        for ext in image_extensions:
            files = list(self.source_dir.glob(ext))
            image_files.extend(files)
            print(f"  {ext}: {len(files)} Dateien gefunden")
        
        # Duplikate entfernen (falls gleiche Datei mit verschiedenen Groß-/Kleinschreibungen)
        image_files = list(set(image_files))
        
        if not image_files:
            print("Keine Bilddateien im Quellordner gefunden!")
            print("Überprüfte Erweiterungen:", image_extensions)
            return
        
        print(f"\nInsgesamt gefunden: {len(image_files)} Bilddateien")
        print(f"Verarbeite alle {len(image_files)} Bilder...")
        print("=" * 50)
        
        successful = 0
        failed = 0
        failed_details = {
            'load_error': 0,
            'no_mask': 0,
            'no_contour': 0,
            'small_area': 0,
            'processing_error': 0
        }
        
        # Fallback-Ordner für fehlgeschlagene Bilder (einfaches Resize)
        fallback_dir = self.output_dir / "fallback_simple_resize"
        fallback_dir.mkdir(exist_ok=True)
        
        for img_file in tqdm(image_files, desc="Pilze ausschneiden"):
            try:
                # Bild laden
                image = cv2.imread(str(img_file))
                if image is None:
                    failed_details['load_error'] += 1
                    failed += 1
                    continue
                
                # Cropping versuchen
                cropped_mushroom = self.crop_mushroom(img_file)
                
                if cropped_mushroom is not None:
                    # Erfolgreich gecroppt
                    output_filename = f"cropped_{img_file.stem}.jpg"
                    output_path = self.output_dir / output_filename
                    cv2.imwrite(str(output_path), cropped_mushroom)
                    successful += 1
                else:
                    # Cropping fehlgeschlagen - Fallback: Einfaches Resize
                    try:
                        # Zentralen Bereich ausschneiden (80% des Bildes)
                        h, w = image.shape[:2]
                        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
                        center_crop = image[margin_h:h-margin_h, margin_w:w-margin_w]
                        
                        # Auf Zielgröße skalieren
                        fallback_resized = cv2.resize(center_crop, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                        
                        # Als Fallback speichern
                        fallback_filename = f"fallback_{img_file.stem}.jpg"
                        fallback_path = fallback_dir / fallback_filename
                        cv2.imwrite(str(fallback_path), fallback_resized)
                        
                        failed_details['no_mask'] += 1
                        failed += 1
                        
                    except Exception as e:
                        failed_details['processing_error'] += 1
                        failed += 1
                        
            except Exception as e:
                failed_details['processing_error'] += 1
                failed += 1
        
        print(f"\nVerarbeitung abgeschlossen:")
        print(f"  Erfolgreich gecroppt: {successful} Bilder")
        print(f"  Fehlgeschlagen: {failed} Bilder")
        print(f"  Erfolgsrate: {(successful/(successful+failed)*100):.1f}%")
        print(f"\nFehlgeschlagene Bilder Details:")
        print(f"  Laden fehlgeschlagen: {failed_details['load_error']}")
        print(f"  Keine Pilzerkennung: {failed_details['no_mask']}")
        print(f"  Verarbeitungsfehler: {failed_details['processing_error']}")
        print(f"\nAusgabe:")
        print(f"  Erfolgreich gecroppt: {self.output_dir.absolute()}")
        print(f"  Fallback (einfach resized): {fallback_dir.absolute()}")
        
        # Empfehlung basierend auf Erfolgsrate
        success_rate = successful/(successful+failed)*100
        if success_rate < 50:
            print(f"\n⚠️  Niedrige Erfolgsrate ({success_rate:.1f}%)!")
            print("   Empfehlungen:")
            print("   - Prüfen Sie die Bildqualität")
            print("   - Möglicherweise sind die Pilze zu klein oder unscharf")
            print("   - Fallback-Bilder wurden mit einfachem Center-Crop erstellt")
        elif success_rate < 80:
            print(f"\n⚠️  Moderate Erfolgsrate ({success_rate:.1f}%)")
            print("   Einige Bilder konnten nicht optimal gecroppt werden")
        else:
            print(f"\n✅ Gute Erfolgsrate ({success_rate:.1f}%)")
        
        return successful, failed
    
    def detect_edges_and_contours(self, image):
        """
        Pilzerkennung vom Zentrum ausgehend - 5% Zentralbereich als Seed
        """
        # Graustufen für Kantenerkennung
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Mehrschichtige Rauschreduktion
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        denoised = cv2.medianBlur(denoised, 3)
        
        # 5% Zentralbereich als garantierter Pilz-Seed
        center_x, center_y = w // 2, h // 2
        seed_radius = int(min(w, h) * 0.05)  # 5% des kleinsten Bilddimension
        
        # Seed-Maske erstellen (zentraler Kreis)
        seed_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(seed_mask, (center_x, center_y), seed_radius, 255, -1)
        
        # Adaptives Watershed vom Zentrum ausgehend
        # 1. Distance Transform für Watershed-Marker
        sure_fg = seed_mask.copy()
        
        # 2. Gradient für Kanten (sanfter als Canny)
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Gradient normalisieren und als "Barriere" für Watershed verwenden
        gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Wasserscheide-Transformation für Region Growing
        # Marker erstellen: Zentrum = 2 (sicherer Vordergrund), Unbekannt = 1
        markers = np.ones_like(gray, dtype=np.int32)
        markers[sure_fg > 0] = 2  # Zentraler Seed
        
        # Watershed anwenden - wächst vom Zentrum aus bis zu starken Kanten
        markers = cv2.watershed(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR), markers)
        
        # Pilzregion extrahieren (Marker 2)
        pilz_mask = np.zeros_like(gray, dtype=np.uint8)
        pilz_mask[markers == 2] = 255
        
        # Morphologische Nachbearbeitung um glatte Konturen zu erhalten
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Kleine Löcher füllen
        pilz_mask = cv2.morphologyEx(pilz_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Konturen glätten
        pilz_mask = cv2.morphologyEx(pilz_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fallback: Wenn Watershed zu restriktiv war, verwende Region Growing basierend auf Farbe
        if np.sum(pilz_mask) < seed_radius * seed_radius * 3.14 * 2:  # Weniger als 2x Seed-Größe
            # Color-basiertes Region Growing vom Zentrum
            pilz_mask = self.region_growing_from_center(denoised, center_x, center_y, seed_radius)
        
        # Konturen finden
        contours, _ = cv2.findContours(pilz_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, pilz_mask
        
        # Größte Kontur wählen (sollte unsere vom Zentrum gewachsene Region sein)
        best_contour = max(contours, key=cv2.contourArea)
        
        # Qualitätsprüfung: Kontur muss den Seed-Bereich enthalten
        seed_point = (center_x, center_y)
        if cv2.pointPolygonTest(best_contour, seed_point, False) < 0:
            # Fallback: Konvexe Hülle um Seed + größte Kontur
            seed_contour = np.array([[center_x-seed_radius, center_y-seed_radius],
                                   [center_x+seed_radius, center_y-seed_radius],
                                   [center_x+seed_radius, center_y+seed_radius],
                                   [center_x-seed_radius, center_y+seed_radius]], dtype=np.int32)
            
            combined_points = np.vstack([best_contour.reshape(-1, 2), seed_contour])
            best_contour = cv2.convexHull(combined_points).reshape(-1, 1, 2)
        
        return best_contour, pilz_mask
    
    def region_growing_from_center(self, gray_image, center_x, center_y, seed_radius):
        """
        Region Growing basierend auf Farbähnlichkeit vom Zentrum ausgehend
        """
        h, w = gray_image.shape
        visited = np.zeros_like(gray_image, dtype=bool)
        result_mask = np.zeros_like(gray_image, dtype=np.uint8)
        
        # Seed-Region im Zentrum
        cv2.circle(result_mask, (center_x, center_y), seed_radius, 255, -1)
        cv2.circle(visited, (center_x, center_y), seed_radius, True, -1)
        
        # Referenzwerte aus Seed-Region
        seed_mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.circle(seed_mask, (center_x, center_y), seed_radius, 255, -1)
        
        reference_values = gray_image[seed_mask > 0]
        mean_intensity = np.mean(reference_values)
        std_intensity = np.std(reference_values)
        tolerance = max(15, std_intensity * 1.5)  # Adaptive Toleranz
        
        # Region Growing mit Floodfill-ähnlichem Ansatz
        # Punkte am Rand der Seed-Region als Startpunkte
        seed_contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not seed_contours:
            return result_mask
        
        # 8-Nachbarschaft
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Warteschlange mit Randpunkten der Seed-Region initialisieren
        queue = []
        for point in seed_contours[0].reshape(-1, 2):
            for dx, dy in directions:
                new_x, new_y = point[0] + dx, point[1] + dy
                if 0 <= new_x < w and 0 <= new_y < h and not visited[new_y, new_x]:
                    queue.append((new_x, new_y))
        
        # Region Growing
        while queue:
            x, y = queue.pop(0)
            
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            current_intensity = gray_image[y, x]
            
            # Prüfen ob Pixel zur Region gehört
            if abs(current_intensity - mean_intensity) <= tolerance:
                result_mask[y, x] = 255
                
                # Nachbarn zur Warteschlange hinzufügen
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < w and 0 <= new_y < h and 
                        not visited[new_y, new_x]):
                        queue.append((new_x, new_y))
        
        return result_mask

    def create_quality_comparison_preview(self, num_samples=3):
        """
        Zeigt Original vs. Objekterkennung vs. Ausgeschnitten (mit schwarzem Hintergrund)
        """
        if not self.source_dir.exists():
            print(f"Quellordner {self.source_dir} existiert nicht!")
            return
        
        # Mehrere Dateierweiterungen suchen
        image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        image_files = []
        
        for ext in image_extensions:
            files = list(self.source_dir.glob(ext))
            image_files.extend(files)
        
        image_files = list(set(image_files))[:num_samples]
        
        if not image_files:
            print("Keine Bilder für Vorschau gefunden!")
            return
        
        preview_dir = self.output_dir / "quality_comparison"
        preview_dir.mkdir(exist_ok=True)
        
        print(f"Erstelle Qualitätsvergleich mit {len(image_files)} Bildern...")
        
        for i, img_file in enumerate(image_files):
            print(f"Analysiere {i+1}/{len(image_files)}: {img_file.name}")
            
            # Original laden
            original = cv2.imread(str(img_file))
            if original is None:
                print(f"Konnte {img_file.name} nicht laden!")
                continue
            
            # Kontrastbasierte Erkennung
            contour, edges = self.detect_edges_and_contours(original)
            
            # Debug-Visualisierung erstellen (Objekterkennung)
            debug_img = original.copy()
            
            if contour is not None:
                # Kontur einzeichnen (grün)
                cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)
                
                # Bounding Box (blau)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Objekterkennung anwenden (mit schwarzem Hintergrund)
                detected_object = self.crop_mushroom(img_file)
            else:
                detected_object = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(detected_object, "NO OBJECT", (50, 112), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Alle auf gleiche Vorschaugröße bringen
            original_small = cv2.resize(original, self.target_size)
            debug_small = cv2.resize(debug_img, self.target_size)
            
            # 3-Panel Layout: Original | Objekterkennung | Ausgeschnitten
            comparison = np.hstack([original_small, debug_small, detected_object])
            
            # Labels hinzufügen
            def add_label(image, text, x, y):
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x-2, y-text_height-5), (x+text_width+2, y+5), (0, 0, 0), -1)
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            add_label(comparison, "Original", 5, 20)
            add_label(comparison, "Objekterkennung", self.target_size[0] + 5, 20)
            add_label(comparison, "Ausgeschnitten", self.target_size[0] * 2 + 5, 20)
            
            # Speichern
            comparison_filename = f"contrast_detection_{i+1}_{img_file.stem}.jpg"
            comparison_path = preview_dir / comparison_filename
            success = cv2.imwrite(str(comparison_path), comparison)
            if success:
                print(f"  Gespeichert: {comparison_filename}")
            else:
                print(f"  Fehler beim Speichern: {comparison_filename}")
        
        print(f"Kontrastbasierte Objekterkennung-Vorschau: {preview_dir.absolute()}")
        
    def create_sample_preview(self, num_samples=5):
        """
        Erstellt eine Vorschau mit ein paar Beispielbildern
        """
        if not self.source_dir.exists():
            print(f"Quellordner {self.source_dir} existiert nicht!")
            return
        
        # Mehrere Dateierweiterungen für Vorschau
        image_extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        image_files = []
        
        for ext in image_extensions:
            files = list(self.source_dir.glob(ext))
            image_files.extend(files)
        
        image_files = list(set(image_files))[:num_samples]
        
        if not image_files:
            print("Keine Bilder für Vorschau gefunden!")
            return
        
        preview_dir = self.output_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        
        print(f"Erstelle Vorschau mit {len(image_files)} Bildern...")
        
        for i, img_file in enumerate(image_files):
            print(f"Verarbeite Vorschau {i+1}/{len(image_files)}: {img_file.name}")
            
            original = cv2.imread(str(img_file))
            cropped = self.crop_mushroom(img_file)
            
            if cropped is not None:
                # Originalbild auch auf Zielgröße skalieren für Vergleich
                original_resized = cv2.resize(original, self.target_size)
                
                # Nebeneinander anzeigen
                comparison = np.hstack([original_resized, cropped])
                
                preview_filename = f"preview_{i+1}_{img_file.stem}.jpg"
                preview_path = preview_dir / preview_filename
                cv2.imwrite(str(preview_path), comparison)
        
        print(f"Vorschau erstellt in: {preview_dir.absolute()}")

def main():
    print("Automatischer Pilz-Cropper")
    print("=" * 30)
    
    # Standard-Pfade
    source = "data/inaturalist_mushrooms/phallus_impudicus"
    target = "data/cropped_mushrooms/Cropped_phallus_impudicus"

    # Benutzer nach Pfaden fragen
    print(f"\nStandardpfade:")
    print(f"  Quelle: {source}")
    print(f"  Ziel: {target}")
    
    use_default = input("\nStandardpfade verwenden? (j/n): ").lower().strip()
    
    if use_default not in ['j', 'ja', 'y', 'yes']:
        source = input("Quellordner eingeben: ").strip()
        target = input("Zielordner eingeben: ").strip()
    
    # Cropper initialisieren
    cropper = MushroomCropper(source, target)
    
    # Vorschau erstellen?
    preview_choice = input("\nVorschau-Optionen:\n1) Einfache Vorschau (j)\n2) Qualitätsvergleich (q)\n3) Keine Vorschau (n)\nWählen Sie: ").lower().strip()
    
    if preview_choice in ['j', 'ja', 'y', 'yes', '1']:
        cropper.create_sample_preview(5)
        
        continue_choice = input("\nMit vollständiger Verarbeitung fortfahren? (j/n): ").lower().strip()
        if continue_choice not in ['j', 'ja', 'y', 'yes']:
            print("Verarbeitung abgebrochen.")
            return
    elif preview_choice in ['q', 'quality', '2']:
        cropper.create_quality_comparison_preview(3)
        
        continue_choice = input("\nQualitätsvergleich erstellt. Mit vollständiger Verarbeitung fortfahren? (j/n): ").lower().strip()
        if continue_choice not in ['j', 'ja', 'y', 'yes']:
            print("Verarbeitung abgebrochen.")
            return
    
    # Alle Bilder verarbeiten
    cropper.process_all_images()

if __name__ == "__main__":
    main()
