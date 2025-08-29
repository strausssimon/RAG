"""
====================================================
Programmname : CountFilesInPath
Beschreibung : Skript zum Zählen von Dateien in Unterordnern. Zeigt die Anzahl der Dateien in data\randomized_mushrooms\inaturalist an

====================================================
"""
import os
from pathlib import Path

def count_files_in_directory(directory_path):
    """
    Zählt alle Bilddateien und alle Dateien in einem Verzeichnis und seinen Unterordnern
    
    Args:
        directory_path (str): Pfad zum Verzeichnis
        
    Returns:
        dict: Dictionary mit Ordnername als Key und (Bilddateien, Gesamtdateien) als Value
    """
    # Unterstützte Bildformate (case-insensitive)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    results = {}
    
    if not os.path.exists(directory_path):
        print(f"Verzeichnis nicht gefunden: {directory_path}")
        return results
    
    # Alle Unterordner finden
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    if not subdirs:
        # Keine Unterordner - zähle Dateien direkt im Hauptordner
        all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
        results["Hauptordner"] = (len(image_files), len(all_files))
    else:
        # Zähle Dateien in jedem Unterordner
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(directory_path, subdir)
            
            # Alle Dateien in diesem Unterordner zählen (rekursiv)
            image_count = 0
            total_count = 0
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    total_count += 1
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in image_extensions:
                        image_count += 1
            
            results[subdir] = (image_count, total_count)
    
    return results

def main():
    """Hauptfunktion"""
    # Bestimme den Basispfad (3 Ebenen nach oben von src/helpers/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..")
    target_dir = os.path.join(base_dir, "data", "inaturalist_mushrooms")
    target_dir = os.path.abspath(target_dir)
    
    print("NEW FILE - iNaturalist Mushrooms Dataset Counter")
    print("=" * 60)
    print(f"ZIELVERZEICHNIS: {target_dir}")
    print("Zählt Bilddateien: .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp")
    print("Zählt alle Dateien: inklusive txt, csv, json, etc.")
    print("=" * 60)
    print()
    
    # Prüfe ob das Verzeichnis existiert
    if not os.path.exists(target_dir):
        print(f"Verzeichnis nicht gefunden: {target_dir}")
        print("Verfügbare Verzeichnisse in data/:")
        
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            for subdir in sorted(subdirs):
                print(f"{subdir}")
        else:
            print("data Verzeichnis nicht gefunden")
        return
    
    # Zähle Dateien
    file_counts = count_files_in_directory(target_dir)
    
    if not file_counts:
        print("Keine Dateien gefunden oder Fehler beim Zugriff")
        return
    
    # Sortiere nach Anzahl der Bilddateien (absteigend)
    sorted_counts = sorted(file_counts.items(), key=lambda x: x[1][0], reverse=True)
    
    print("Ergebnisse (sortiert nach Anzahl der Bilddateien):")
    print("-" * 80)
    
    total_images = 0
    total_files = 0
    
    for folder_name, (image_count, total_count) in sorted_counts:
        total_images += image_count
        total_files += total_count
        print(f"{folder_name:<30} | {image_count:>6} Bilddateien | {total_count:>6} Dateien gesamt")
    
    print("-" * 80)
    print(f"Gesamt: {total_images:>6} Bilddateien | {total_files:>6} Dateien gesamt")
    
    if len(file_counts) > 1:
        # Berechne Statistiken nur wenn mehr als ein Ordner
        avg_images = total_images / len(file_counts)
        avg_files = total_files / len(file_counts)
        
        image_counts = [count[0] for count in file_counts.values()]
        file_counts_total = [count[1] for count in file_counts.values()]
        
        max_images = max(image_counts)
        min_images = min(image_counts)
        max_files = max(file_counts_total)
        min_files = min(file_counts_total)
        
        # Finde Ordner mit max/min
        max_folder_images = max(file_counts, key=lambda x: file_counts[x][0])
        min_folder_images = min(file_counts, key=lambda x: file_counts[x][0])
        
        print("Statistiken:")
        print(f"   Durchschnitt pro Ordner: {avg_images:.1f} Bilddateien | {avg_files:.1f} Dateien gesamt")
        print(f"   Maximum: {max_images} Bilddateien | {max_files} Dateien gesamt ({max_folder_images})")
        print(f"   Minimum: {min_images} Bilddateien | {min_files} Dateien gesamt ({min_folder_images})")
        print(f"   Anzahl Ordner: {len(file_counts)}")

if __name__ == "__main__":
    main()
