#!/usr/bin/env python3
"""
Skript zum Zählen von Dateien in Unterordnern
Zeigt die Anzahl der Dateien in data\augmented_mushrooms\resized an
"""

import os
import glob
from pathlib import Path

def count_files_in_directory(directory_path):
    """
    Zählt alle Bilddateien (jpg, jpeg, png, bmp, gif, tiff, webp) in einem Verzeichnis und seinen Unterordnern
    
    Args:
        directory_path (str): Pfad zum Verzeichnis
        
    Returns:
        dict: Dictionary mit Ordnername als Key und Bilddateianzahl als Value
    """
    # Unterstützte Bildformate (case-insensitive)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    results = {}
    
    if not os.path.exists(directory_path):
        print(f"❌ Verzeichnis nicht gefunden: {directory_path}")
        return results
    
    # Alle Unterordner finden
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    if not subdirs:
        # Keine Unterordner - zähle Bilddateien direkt im Hauptordner
        files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f)) 
                and os.path.splitext(f)[1].lower() in image_extensions]
        results["Hauptordner"] = len(files)
    else:
        # Zähle Bilddateien in jedem Unterordner
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(directory_path, subdir)
            
            # Alle Bilddateien in diesem Unterordner zählen (rekursiv)
            image_count = 0
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in image_extensions:
                        image_count += 1
            
            results[subdir] = image_count
    
    return results

def main():
    """Hauptfunktion"""
    # Bestimme den Basispfad (3 Ebenen nach oben von src/helpers/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..")
    target_dir = os.path.join(base_dir, "data", "all_mushrooms")
    target_dir = os.path.abspath(target_dir)
    
    print("🔍 Bilddateien-Zähler für Pilz-Dataset")
    print("=" * 50)
    print(f"Zielverzeichnis: {target_dir}")
    print("📷 Zählt nur Bilddateien: .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp")
    print()
    
    # Prüfe ob das Verzeichnis existiert
    if not os.path.exists(target_dir):
        print(f"❌ Verzeichnis nicht gefunden: {target_dir}")
        print("Verfügbare Verzeichnisse in data/:")
        
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            for subdir in sorted(subdirs):
                print(f"  📁 {subdir}")
        else:
            print("  ❌ data Verzeichnis nicht gefunden")
        return
    
    # Zähle Bilddateien
    file_counts = count_files_in_directory(target_dir)
    
    if not file_counts:
        print("❌ Keine Bilddateien gefunden oder Fehler beim Zugriff")
        return
    
    # Ergebnisse anzeigen
    print("📊 Ergebnisse (sortiert nach Anzahl der Bilddateien):")
    print("-" * 50)
    
    # Sortiere nach Anzahl der Bilddateien (absteigend)
    sorted_counts = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    
    total_files = 0
    for folder_name, count in sorted_counts:
        print(f"📁 {folder_name:30} | {count:6} Bilddateien")
        total_files += count
    
    print("-" * 50)
    print(f"📋 Gesamt:                      | {total_files:6} Bilddateien")
    print()
    
    # Zusätzliche Statistiken
    if len(file_counts) > 1:
        avg_files = total_files / len(file_counts)
        max_files = max(file_counts.values())
        min_files = min(file_counts.values())
        
        # Finde Ordner mit max/min Bilddateien
        max_folder = max(file_counts, key=file_counts.get)
        min_folder = min(file_counts, key=file_counts.get)
        
        print("📈 Statistiken:")
        print(f"   Durchschnitt pro Ordner: {avg_files:.1f} Bilddateien")
        print(f"   Maximum: {max_files} Bilddateien ({max_folder})")
        print(f"   Minimum: {min_files} Bilddateien ({min_folder})")
        print(f"   Anzahl Ordner: {len(file_counts)}")

if __name__ == "__main__":
    main()
