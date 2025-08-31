#!/usr/bin/env python3
"""
Skript zum Entfernen von Bildern mit zu geringer Auflösung
Entfernt alle Bilder deren Breite ODER Höhe kleiner als 250 Pixel ist
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def remove_small_images(root_dir="data/randomized_mushrooms/inaturalist", min_size=200, dry_run=True):
    """
    Entfernt Bilder mit Breite oder Höhe kleiner als min_size
    
    Args:
        root_dir (str): Pfad zum Wurzelverzeichnis
        min_size (int): Minimale Breite/Höhe in Pixeln
        dry_run (bool): Wenn True, werden nur die Dateien angezeigt, aber nicht gelöscht
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Fehler: Ordner {root_path} existiert nicht!")
        return
    
    print(f"Durchsuche Verzeichnis: {root_path.absolute()}")
    print(f"Mindestgröße: {min_size}x{min_size} Pixel")
    print(f"Modus: {'DRY RUN (nur anzeigen)' if dry_run else 'LÖSCHEN AKTIVIERT'}")
    print("=" * 80)
    
    # Statistiken
    total_images = 0
    small_images = 0
    corrupted_images = 0
    deleted_files = []
    
    # Unterstützte Bildformate
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Alle Bilddateien finden (rekursiv)
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(list(root_path.rglob(f"*{ext}")))
        all_image_files.extend(list(root_path.rglob(f"*{ext.upper()}")))
    
    print(f"Gefundene Bilddateien: {len(all_image_files)}")
    print()
    
    # Alle Bilder durchgehen
    for image_path in tqdm(all_image_files, desc="Analysiere Bilder"):
        total_images += 1
        
        try:
            # Lade Bild mit PIL (robuster für verschiedene Formate)
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Prüfe ob Bild zu klein ist
                if width < min_size or height < min_size:
                    small_images += 1
                    
                    relative_path = image_path.relative_to(root_path)
                    print(f"❌ KLEIN: {relative_path} ({width}x{height})")
                    
                    deleted_files.append({
                        'path': str(image_path),
                        'relative_path': str(relative_path),
                        'size': f"{width}x{height}",
                        'width': width,
                        'height': height
                    })
                    
                    # Lösche Datei wenn nicht im Dry-Run Modus
                    if not dry_run:
                        try:
                            os.remove(image_path)
                            print(f"  → GELÖSCHT")
                        except Exception as delete_error:
                            print(f"  → FEHLER beim Löschen: {delete_error}")
                    
        except Exception as e:
            corrupted_images += 1
            relative_path = image_path.relative_to(root_path)
            print(f"💥 KORRUPT: {relative_path} - {e}")
            
            # Korrupte Bilder auch löschen
            if not dry_run:
                try:
                    os.remove(image_path)
                    print(f"  → KORRUPTE DATEI GELÖSCHT")
                except Exception as delete_error:
                    print(f"  → FEHLER beim Löschen: {delete_error}")
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("📊 ZUSAMMENFASSUNG:")
    print(f"   Gesamte Bilder analysiert: {total_images}")
    print(f"   Zu kleine Bilder gefunden: {small_images}")
    print(f"   Korrupte Bilder gefunden: {corrupted_images}")
    print(f"   Problematische Bilder gesamt: {small_images + corrupted_images}")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN - Keine Dateien wurden gelöscht!")
        print(f"   Zum tatsächlichen Löschen: dry_run=False setzen")
    else:
        print(f"\n✅ LÖSCHEN ABGESCHLOSSEN!")
        print(f"   {small_images + corrupted_images} Dateien wurden entfernt")
    
    # Detaillierte Liste der zu löschenden/gelöschten Dateien
    if deleted_files:
        print(f"\n📋 DETAILLIERTE LISTE ({len(deleted_files)} Dateien):")
        print("-" * 80)
        
        # Sortiere nach Größe (kleinste zuerst)
        deleted_files.sort(key=lambda x: x['width'] * x['height'])
        
        for file_info in deleted_files:
            print(f"  {file_info['size']:<12} | {file_info['relative_path']}")
    
    print("\n" + "=" * 80)
    
    return {
        'total_images': total_images,
        'small_images': small_images,
        'corrupted_images': corrupted_images,
        'deleted_files': deleted_files
    }

def main():
    """Hauptfunktion"""
    print("🖼️  BILD-BEREINIGUNG: Entfernung kleiner Bilder")
    print("=" * 60)
    
    # Konfiguration
    root_directory = "data/randomized_mushrooms/inaturalist"
    minimum_size = 200
    
    print(f"Zielverzeichnis: {root_directory}")
    print(f"Mindestgröße: {minimum_size}x{minimum_size} Pixel")
    print("Kriterium: Breite ODER Höhe < 200px → Löschen")
    print()
    
    # Erste Analyse im Dry-Run Modus
    print("🔍 SCHRITT 1: Analyse (Dry Run)")
    print("-" * 40)
    
    stats = remove_small_images(
        root_dir=root_directory,
        min_size=minimum_size,
        dry_run=True
    )
    
    if stats['small_images'] > 0 or stats['corrupted_images'] > 0:
        print("\n" + "⚠️" * 20)
        print("WARNUNG: Problematische Bilder gefunden!")
        print("⚠️" * 20)
        
        user_input = input("\nMöchten Sie diese Dateien wirklich löschen? (ja/nein): ").strip().lower()
        
        if user_input in ['ja', 'j', 'yes', 'y']:
            print("\n🗑️  SCHRITT 2: Löschen aktiviert")
            print("-" * 40)
            
            final_stats = remove_small_images(
                root_dir=root_directory,
                min_size=minimum_size,
                dry_run=False
            )
            
            print(f"\n🎉 BEREINIGUNG ABGESCHLOSSEN!")
            print(f"   {final_stats['small_images'] + final_stats['corrupted_images']} Dateien entfernt")
        else:
            print("\n❌ Abgebrochen - Keine Dateien wurden gelöscht")
    else:
        print("\n✅ Alle Bilder haben ausreichende Größe!")
        print("   Keine Aktion erforderlich")

if __name__ == "__main__":
    main()
