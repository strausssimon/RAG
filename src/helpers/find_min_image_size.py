import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image

def find_minimum_image_size(root_dir="Webscraper/data/images_mushrooms"):
    """Findet die kleinste Bildgröße in allen Unterordnern"""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Fehler: Ordner {root_path} existiert nicht!")
        return
    
    print(f"Suche nach kleinster Bildgröße in: {root_path.absolute()}")
    print("=" * 60)
    
    min_width = float('inf')
    min_height = float('inf')
    min_area = float('inf')
    smallest_image_path = None
    smallest_area_path = None
    
    total_images = 0
    corrupted_images = 0
    
    # Statistiken sammeln
    all_widths = []
    all_heights = []
    all_areas = []
    
    # Alle Unterordner durchgehen
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\nAnalysiere Klasse: {class_name}")
        
        # Alle Bildformate finden
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))
            image_files.extend(list(class_dir.glob(ext.upper())))
        
        if not image_files:
            print(f"   Keine Bilder gefunden")
            continue
        
        class_min_width = float('inf')
        class_min_height = float('inf')
        class_images_count = 0
        
        # Jedes Bild analysieren
        for img_file in image_files:
            try:
                # Versuche mit PIL zu laden (schneller für Größeninfo)
                with Image.open(img_file) as img:
                    width, height = img.size
                    area = width * height
                    
                    # Statistiken aktualisieren
                    all_widths.append(width)
                    all_heights.append(height)
                    all_areas.append(area)
                    
                    # Minimale Breite/Höhe
                    if width < min_width:
                        min_width = width
                        smallest_image_path = img_file
                    
                    if height < min_height:
                        min_height = height
                        if smallest_image_path != img_file:
                            smallest_image_path = img_file
                    
                    # Kleinste Fläche
                    if area < min_area:
                        min_area = area
                        smallest_area_path = img_file
                    
                    # Klassen-Minimums
                    class_min_width = min(class_min_width, width)
                    class_min_height = min(class_min_height, height)
                    
                    class_images_count += 1
                    total_images += 1
                    
            except Exception as e:
                print(f"   Fehler bei {img_file.name}: {e}")
                corrupted_images += 1
        
        if class_images_count > 0:
            print(f"   {class_images_count} Bilder analysiert")
            print(f"   Kleinste Breite in Klasse: {class_min_width}px")
            print(f"   Kleinste Höhe in Klasse: {class_min_height}px")
    
    # Ergebnisse anzeigen
    print("\n" + "=" * 60)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 60)
    
    if total_images == 0:
        print("Keine gültigen Bilder gefunden!")
        return
    
    print(f"Analysierte Bilder: {total_images}")
    print(f"Beschädigte Bilder: {corrupted_images}")
    print()
    
    print("MINIMALE GRÖSSEN:")
    print(f"Kleinste Breite: {min_width}px")
    print(f"Kleinste Höhe: {min_height}px")
    print(f"Kleinste Fläche: {min_area}px² ({int(np.sqrt(min_area))}x{int(np.sqrt(min_area))} als Quadrat)")
    print()
    
    if smallest_image_path:
        print(f"Bild mit kleinster Dimension: {smallest_image_path}")
    
    if smallest_area_path:
        print(f"Bild mit kleinster Fläche: {smallest_area_path}")
    
    # Statistiken
    print("\nSTATISTIKEN:")
    print(f"Durchschnittliche Breite: {np.mean(all_widths):.1f}px")
    print(f"Durchschnittliche Höhe: {np.mean(all_heights):.1f}px")
    print(f"Durchschnittliche Fläche: {np.mean(all_areas):.0f}px²")
    print()
    
    print(f"Median Breite: {np.median(all_widths):.1f}px")
    print(f"Median Höhe: {np.median(all_heights):.1f}px")
    print(f"Median Fläche: {np.median(all_areas):.0f}px²")
    print()
    
    print("GRÖSSEN-VERTEILUNG:")
    # Erstelle Größenkategorien
    size_categories = {
        "Sehr klein (< 100px)": 0,
        "Klein (100-300px)": 0,
        "Mittel (300-600px)": 0,
        "Groß (600-1000px)": 0,
        "Sehr groß (> 1000px)": 0
    }
    
    for width, height in zip(all_widths, all_heights):
        max_dim = max(width, height)
        if max_dim < 100:
            size_categories["Sehr klein (< 100px)"] += 1
        elif max_dim < 300:
            size_categories["Klein (100-300px)"] += 1
        elif max_dim < 600:
            size_categories["Mittel (300-600px)"] += 1
        elif max_dim < 1000:
            size_categories["Groß (600-1000px)"] += 1
        else:
            size_categories["Sehr groß (> 1000px)"] += 1
    
    for category, count in size_categories.items():
        percentage = (count / total_images) * 100
        print(f"{category}: {count} Bilder ({percentage:.1f}%)")
    
    # Empfehlung für Resize-Größe
    print("\nEMPFEHLUNG:")
    recommended_size = min(128, min_width, min_height)
    print(f"Empfohlene einheitliche Größe: {recommended_size}x{recommended_size}px")
    print(f"(Basierend auf kleinster Dimension: {min(min_width, min_height)}px)")
    
    # Detaillierte Info über das kleinste Bild
    if smallest_area_path:
        try:
            with Image.open(smallest_area_path) as img:
                width, height = img.size
                print(f"\nDETAILS ZUM KLEINSTEN BILD:")
                print(f"Datei: {smallest_area_path}")
                print(f"Größe: {width}x{height}px")
                print(f"Fläche: {width * height}px²")
                print(f"Seitenverhältnis: {width/height:.2f}")
        except:
            pass

def quick_size_check():
    """Schnelle Überprüfung der ersten paar Bilder jeder Klasse"""
    root_path = Path("Webscraper/data/images_mushrooms")
    
    print("SCHNELLE GRÖSSEN-ÜBERPRÜFUNG")
    print("=" * 40)
    
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        # Ersten 3 Bilder jeder Klasse prüfen
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))
            image_files.extend(list(class_dir.glob(ext.upper())))
        
        if image_files:
            print(f"\n{class_dir.name}:")
            for i, img_file in enumerate(image_files[:3]):
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        print(f"  {img_file.name}: {width}x{height}px")
                except:
                    print(f"  {img_file.name}: Fehler beim Laden")

if __name__ == "__main__":
    print("🔍 BILDGRÖSSEN-ANALYSE 🔍")
    print("=" * 50)
    
    choice = input("Vollständige Analyse (f) oder schnelle Überprüfung (s)? [f/s]: ").lower().strip()
    
    if choice == 's':
        quick_size_check()
    else:
        find_minimum_image_size()
