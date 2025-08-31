"""
Test Model Data Extractor

Dieses Skript findet Bilder, die in data/inaturalist_mushrooms vorhanden sind,
aber NICHT in data/randomized_mushrooms/inaturalist, und kopiert sie nach
data/test_mushrooms_randomized/inaturalist.

Zweck: Echte, unabhängige Testdaten erstellen, die nicht im Training verwendet wurden.

Beispiel:
- data/inaturalist_mushrooms/Boletus_edulis: 2648 Bilder
- data/randomized_mushrooms/inaturalist/Boletus_edulis: 1500 Bilder (Subset)
- Ergebnis: 1148 Bilder (2648 - 1500) werden nach data/test_mushrooms_randomized/inaturalist/Boletus_edulis kopiert
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

class TestDataExtractor:
    def __init__(self):
        self.source_dir = Path("data/inaturalist_mushrooms")
        self.randomized_dir = Path("data/randomized_mushrooms/inaturalist")
        self.output_dir = Path("data/test_mushrooms_randomized/inaturalist")
        
        # Klassen, die verarbeitet werden sollen (5 Hauptklassen)
        self.target_classes = [
            "Phallus_impudicus", "Amanita_muscaria", "Boletus_edulis", 
            "Cantharellus_cibarius", "Armillaria_mellea"
        ]
        
        # Unterstützte Bildformate
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
    def get_image_files(self, directory):
        """Sammelt alle Bilddateien aus einem Verzeichnis"""
        image_files = set()
        
        if not directory.exists():
            return image_files
            
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                # Verwende nur den Dateinamen (ohne Pfad) für Vergleich
                image_files.add(file_path.name.lower())
                
        return image_files
    
    def get_full_file_paths(self, directory):
        """Sammelt alle Bilddateien mit vollständigen Pfaden"""
        file_dict = {}
        
        if not directory.exists():
            return file_dict
            
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                file_dict[file_path.name.lower()] = file_path
                
        return file_dict
    
    def extract_test_data(self, dry_run=True):
        """
        Extrahiert Testdaten für alle Klassen
        
        Args:
            dry_run: Wenn True, wird nur analysiert aber nicht kopiert
        """
        print("🍄 TEST DATA EXTRACTOR - Mushroom Dataset")
        print("=" * 60)
        print(f"Quelle: {self.source_dir}")
        print(f"Bereits verwendet: {self.randomized_dir}")
        print(f"Ziel für Testdaten: {self.output_dir}")
        print("=" * 60)
        
        if not self.source_dir.exists():
            print(f"❌ Fehler: Quellverzeichnis {self.source_dir} existiert nicht!")
            return
            
        if not self.randomized_dir.exists():
            print(f"❌ Fehler: Randomized-Verzeichnis {self.randomized_dir} existiert nicht!")
            return
            
        # Erstelle Ausgabeverzeichnis falls es nicht existiert
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_found = 0
        total_copied = 0
        class_stats = defaultdict(dict)
        
        print(f"\nAnalysiere {len(self.target_classes)} Klassen...")
        
        for class_name in self.target_classes:
            print(f"\n📂 Verarbeite Klasse: {class_name}")
            
            source_class_dir = self.source_dir / class_name
            randomized_class_dir = self.randomized_dir / class_name
            output_class_dir = self.output_dir / class_name
            
            if not source_class_dir.exists():
                print(f"   ⚠️ Klasse {class_name} nicht in Quelle gefunden!")
                continue
                
            # Sammle Dateinamen aus beiden Verzeichnissen
            print("   📊 Analysiere Dateien...")
            source_files = self.get_full_file_paths(source_class_dir)
            randomized_files = self.get_image_files(randomized_class_dir)
            
            print(f"   📈 Quelle: {len(source_files)} Bilder")
            print(f"   📈 Bereits verwendet: {len(randomized_files)} Bilder")
            
            # Finde Bilder, die in der Quelle aber nicht in randomized sind
            unique_files = []
            for filename, filepath in source_files.items():
                if filename not in randomized_files:
                    unique_files.append(filepath)
            
            print(f"   ✨ Neue Testbilder gefunden: {len(unique_files)}")
            
            # Statistiken sammeln
            class_stats[class_name] = {
                'source_count': len(source_files),
                'randomized_count': len(randomized_files),
                'test_count': len(unique_files)
            }
            
            total_found += len(unique_files)
            
            if not dry_run and unique_files:
                # Erstelle Klassenverzeichnis
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Kopiere Dateien
                print(f"   📁 Kopiere nach: {output_class_dir}")
                copied_count = 0
                
                for source_file in tqdm(unique_files, desc=f"Kopiere {class_name}"):
                    try:
                        dest_file = output_class_dir / source_file.name
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                    except Exception as e:
                        print(f"   ❌ Fehler beim Kopieren von {source_file.name}: {e}")
                
                print(f"   ✅ Erfolgreich kopiert: {copied_count} Dateien")
                total_copied += copied_count
            elif unique_files:
                print(f"   🔍 Würde {len(unique_files)} Dateien kopieren (DRY RUN)")
        
        # Zusammenfassung
        print("\n" + "=" * 60)
        print("📊 ZUSAMMENFASSUNG")
        print("=" * 60)
        
        for class_name, stats in class_stats.items():
            if stats['test_count'] > 0:
                percentage = (stats['test_count'] / stats['source_count']) * 100 if stats['source_count'] > 0 else 0
                print(f"{class_name}:")
                print(f"  📈 Quelle: {stats['source_count']} | Verwendet: {stats['randomized_count']} | Test: {stats['test_count']} ({percentage:.1f}%)")
        
        print(f"\n🎯 GESAMT GEFUNDEN: {total_found} neue Testbilder")
        
        if not dry_run:
            print(f"✅ GESAMT KOPIERT: {total_copied} Dateien")
            print(f"📁 Ziel: {self.output_dir.absolute()}")
        else:
            print("🔍 DRY RUN - Keine Dateien wurden kopiert")
            print("💡 Verwende dry_run=False zum tatsächlichen Kopieren")
        
        return class_stats
    
    def verify_extraction(self):
        """Überprüft die extrahierten Testdaten"""
        print("\n🔍 VERIFIKATION der extrahierten Testdaten")
        print("=" * 50)
        
        if not self.output_dir.exists():
            print("❌ Keine Testdaten gefunden!")
            return
            
        total_test_files = 0
        
        for class_name in self.target_classes:
            test_class_dir = self.output_dir / class_name
            
            if test_class_dir.exists():
                test_files = list(test_class_dir.glob("*"))
                test_files = [f for f in test_files if f.suffix.lower() in self.image_extensions]
                print(f"{class_name}: {len(test_files)} Testbilder")
                total_test_files += len(test_files)
            else:
                print(f"{class_name}: Kein Verzeichnis gefunden")
        
        print(f"\n✅ GESAMT: {total_test_files} Testbilder verfügbar")

def main():
    extractor = TestDataExtractor()
    
    print("SCHRITT 1: Analyse (Dry Run)")
    print("=" * 40)
    stats = extractor.extract_test_data(dry_run=True)
    
    if stats:
        print("\n" + "=" * 40)
        choice = input("Möchten Sie die Testdaten extrahieren? (j/n): ").lower().strip()
        
        if choice in ['j', 'ja', 'y', 'yes']:
            print("\nSCHRITT 2: Extraktion")
            print("=" * 40)
            extractor.extract_test_data(dry_run=False)
            
            print("\nSCHRITT 3: Verifikation")
            print("=" * 40)
            extractor.verify_extraction()
        else:
            print("❌ Extraktion abgebrochen.")
    else:
        print("❌ Keine Daten zum Extrahieren gefunden.")

if __name__ == "__main__":
    main()