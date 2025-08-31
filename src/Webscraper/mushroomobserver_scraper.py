#!/usr/bin/env python3
"""
Mushroom Image Scraper
Lädt Bilder aus der Excel-Datei mushroom_data.xlsx herunter und organisiert sie nach Pilznamen
"""

import os
import pandas as pd
import requests
import time
import re
from urllib.parse import urlparse
from pathlib import Path
import logging
from typing import Optional, Tuple

# Logging konfigurieren mit UTF-8 Encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mushroom_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setze explizit UTF-8 für StreamHandler falls möglich
try:
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except:
    pass

class MushroomImageScraper:
    def __init__(self, excel_file: str, output_dir: str = "downloaded_mushrooms"):
        """
        Initialisiert den Mushroom Image Scraper
        
        Args:
            excel_file (str): Pfad zur Excel-Datei
            output_dir (str): Zielverzeichnis für heruntergeladene Bilder
        """
        self.excel_file = excel_file
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Gewünschte Pilzarten (deutsche und lateinische Namen)
        self.target_mushrooms = {
            # Deutsche Namen -> Lateinische Namen (Mapping)
            'steinpilz': ['boletus_edulis', 'boletus edulis'],
            'fliegenpilz': ['amanita_muscaria', 'amanita muscaria'],
            'hexenröhrling': ['boletus_erythropus', 'boletus erythropus', 'suillellus_luridus', 'suillellus luridus'],
            'pfifferling': ['cantharellus_cibarius', 'cantharellus cibarius'],
            'hallimasche': ['armillaria_mellea', 'armillaria mellea', 'armillaria_ostoyae', 'armillaria ostoyae'],
            'waldegerling': ['agaricus_silvaticus', 'agaricus silvaticus'],
            'bitterling': ['tylopilus_felleus', 'tylopilus felleus'],
            'maronen-röhrling': ['imleria_badia', 'imleria badia', 'boletus_badius', 'boletus badius'],
            'pantherpilz': ['amanita_pantherina', 'amanita pantherina'],
            'knollenblätterpilz': ['amanita_phalloides', 'amanita phalloides']
        }
        
        # Statistiken
        self.stats = {
            'total_images': 0,
            'filtered_images': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'folders_created': 0
        }
    
    def is_target_mushroom(self, mushroom_name: str) -> bool:
        """
        Prüft ob ein Pilzname zu den gewünschten Pilzarten gehört
        
        Args:
            mushroom_name (str): Name des Pilzes
            
        Returns:
            bool: True wenn der Pilz gewünscht ist, False sonst
        """
        name_lower = mushroom_name.lower().strip()
        
        # Ersetze häufige Variationen
        name_normalized = name_lower.replace('-', ' ').replace('_', ' ')
        
        # Prüfe gegen alle Ziel-Pilze
        for german_name, latin_variants in self.target_mushrooms.items():
            # Prüfe deutschen Namen
            if german_name in name_normalized:
                return True
            
            # Prüfe lateinische Namen
            for latin_name in latin_variants:
                latin_normalized = latin_name.replace('_', ' ')
                if latin_normalized in name_normalized:
                    return True
                    
                # Auch einzelne Wörter prüfen (z.B. "boletus" oder "amanita")
                latin_words = latin_normalized.split()
                if len(latin_words) >= 2:
                    genus = latin_words[0]  # Gattung
                    species = latin_words[1]  # Art
                    if genus in name_normalized and species in name_normalized:
                        return True
        
        return False
    
    def get_target_folder_name(self, mushroom_name: str) -> str:
        """
        Bestimmt den Zielordnernamen für einen Pilz
        
        Args:
            mushroom_name (str): Name des Pilzes
            
        Returns:
            str: Standardisierter Ordnername
        """
        name_lower = mushroom_name.lower().strip()
        name_normalized = name_lower.replace('-', ' ').replace('_', ' ')
        
        # Mappe auf deutsche Standardnamen
        for german_name, latin_variants in self.target_mushrooms.items():
            # Prüfe deutschen Namen
            if german_name in name_normalized:
                return german_name.replace('-', '_')
            
            # Prüfe lateinische Namen
            for latin_name in latin_variants:
                latin_normalized = latin_name.replace('_', ' ')
                if latin_normalized in name_normalized:
                    return german_name.replace('-', '_')
                    
                # Auch einzelne Wörter prüfen
                latin_words = latin_normalized.split()
                if len(latin_words) >= 2:
                    genus = latin_words[0]
                    species = latin_words[1]
                    if genus in name_normalized and species in name_normalized:
                        return german_name.replace('-', '_')
        
        # Fallback: bereinigter ursprünglicher Name
        return self.sanitize_filename(mushroom_name)

    def sanitize_filename(self, name: str) -> str:
        """
        Bereinigt einen Namen für die Verwendung als Datei-/Ordnername
        
        Args:
            name (str): Ursprünglicher Name
            
        Returns:
            str: Bereinigter Name
        """
        # Kleinbuchstaben und Leerzeichen durch Unterstriche ersetzen
        sanitized = re.sub(r'[^\w\s-]', '', name.strip())
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.lower()
    
    def extract_image_id(self, image_url: str) -> Optional[str]:
        """
        Extrahiert die Bild-ID aus der URL
        
        Args:
            image_url (str): Bild-URL
            
        Returns:
            Optional[str]: Bild-ID oder None
        """
        try:
            # Beispiel: https://mushroomobserver.org/images/640/1752672.jpg
            # Extrahiere die Nummer vor der Dateiendung
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            image_id = os.path.splitext(filename)[0]
            return image_id
        except Exception as e:
            logger.warning(f"Konnte Bild-ID nicht extrahieren von {image_url}: {e}")
            return None
    
    def create_folder(self, folder_name: str) -> str:
        """
        Erstellt einen Ordner für einen Pilznamen
        
        Args:
            folder_name (str): Name des Ordners
            
        Returns:
            str: Pfad zum erstellten Ordner
        """
        folder_path = os.path.join(self.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """
        Lädt ein Bild herunter
        
        Args:
            image_url (str): URL des Bildes
            save_path (str): Pfad zum Speichern
            
        Returns:
            bool: True wenn erfolgreich, False sonst
        """
        try:
            # Prüfe ob Datei bereits existiert
            if os.path.exists(save_path):
                logger.info(f"Datei existiert bereits: {os.path.basename(save_path)}")
                self.stats['skipped'] += 1
                return True
            
            # Lade Bild herunter
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Speichere Bild
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"[OK] Heruntergeladen: {os.path.basename(save_path)}")
            self.stats['downloaded'] += 1
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[FEHLER] Beim Herunterladen {image_url}: {e}")
            self.stats['failed'] += 1
            return False
        except Exception as e:
            logger.error(f"[FEHLER] Unerwarteter Fehler beim Herunterladen {image_url}: {e}")
            self.stats['failed'] += 1
            return False
    
    def process_excel_data(self) -> pd.DataFrame:
        """
        Lädt und verarbeitet die Excel-Daten
        
        Returns:
            pd.DataFrame: Verarbeitete Daten (nur gewünschte Pilzarten)
        """
        try:
            # Lade Excel-Datei
            logger.info(f"Lade Excel-Datei: {self.excel_file}")
            df = pd.read_excel(self.excel_file)
            
            # Prüfe ob erforderliche Spalten vorhanden sind
            required_columns = ['name', 'image']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Fehlende Spalten in Excel-Datei: {missing_columns}")
            
            # Entferne Zeilen mit leeren Werten
            df = df.dropna(subset=required_columns)
            self.stats['total_images'] = len(df)
            logger.info(f"Gesamt Einträge in Excel: {len(df)}")
            
            # Filtere nur gewünschte Pilzarten
            logger.info("Filtere gewünschte Pilzarten...")
            filtered_df = df[df['name'].apply(self.is_target_mushroom)]
            self.stats['filtered_images'] = len(filtered_df)
            
            logger.info(f"Gefilterte Einträge (gewünschte Pilze): {len(filtered_df)}")
            
            # Zeige gefundene Pilzarten
            if len(filtered_df) > 0:
                found_mushrooms = filtered_df['name'].unique()
                logger.info("Gefundene Pilzarten:")
                for mushroom in sorted(found_mushrooms):
                    count = len(filtered_df[filtered_df['name'] == mushroom])
                    target_folder = self.get_target_folder_name(mushroom)
                    logger.info(f"  [PILZ] {mushroom} -> {target_folder} ({count} Bilder)")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Excel-Datei: {e}")
            raise
    
    def run(self):
        """
        Führt den Scraping-Prozess aus
        """
        logger.info("*** Mushroom Image Scraper gestartet ***")
        logger.info("=" * 50)
        
        try:
            # Erstelle Ausgabeordner
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Ausgabeordner: {os.path.abspath(self.output_dir)}")
            
            # Lade Excel-Daten
            df = self.process_excel_data()
            
            # Verarbeite jeden Eintrag
            created_folders = set()
            
            for index, row in df.iterrows():
                mushroom_name = str(row['name']).strip()
                image_url = str(row['image']).strip()
                
                # Überspringe ungültige URLs
                if not image_url.startswith('http'):
                    logger.warning(f"Ungültige URL übersprungen: {image_url}")
                    self.stats['failed'] += 1
                    continue
                
                # Bestimme Zielordner basierend auf Pilzname
                folder_name = self.get_target_folder_name(mushroom_name)
                
                # Erstelle Ordner falls nötig
                if folder_name not in created_folders:
                    folder_path = self.create_folder(folder_name)
                    created_folders.add(folder_name)
                    self.stats['folders_created'] += 1
                    logger.info(f"[ORDNER] Erstellt: {folder_name}")
                else:
                    folder_path = os.path.join(self.output_dir, folder_name)
                
                # Extrahiere Bild-ID
                image_id = self.extract_image_id(image_url)
                if not image_id:
                    image_id = f"unknown_{index}"
                
                # Erstelle Dateiname
                file_extension = os.path.splitext(urlparse(image_url).path)[1]
                if not file_extension:
                    file_extension = '.jpg'  # Fallback
                
                filename = f"{folder_name}_{image_id}{file_extension}"
                save_path = os.path.join(folder_path, filename)
                
                # Lade Bild herunter
                logger.info(f"Verarbeite ({index+1}/{len(df)}): {mushroom_name} -> {filename}")
                self.download_image(image_url, save_path)
                
                # Kurze Pause zwischen Downloads
                time.sleep(0.5)
            
            # Zeige Statistiken
            self.show_statistics()
            
        except Exception as e:
            logger.error(f"Fehler beim Ausführen des Scrapers: {e}")
            raise
    
    def show_statistics(self):
        """
        Zeigt Download-Statistiken an
        """
        logger.info("\n" + "=" * 50)
        logger.info("*** DOWNLOAD-STATISTIKEN ***")
        logger.info("=" * 50)
        logger.info(f"Gesamt Bilder in Excel:   {self.stats['total_images']}")
        logger.info(f"Gefilterte Bilder:        {self.stats['filtered_images']}")
        logger.info(f"Heruntergeladen:          {self.stats['downloaded']}")
        logger.info(f"Übersprungen:             {self.stats['skipped']}")
        logger.info(f"Fehlgeschlagen:           {self.stats['failed']}")
        logger.info(f"Ordner erstellt:          {self.stats['folders_created']}")
        
        if self.stats['filtered_images'] > 0:
            success_rate = (self.stats['downloaded'] / self.stats['filtered_images'] * 100)
            logger.info(f"Erfolgsrate (gefiltert):  {success_rate:.1f}%")
        
        filter_rate = (self.stats['filtered_images'] / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
        logger.info(f"Filterrate:               {filter_rate:.1f}%")
        logger.info("=" * 50)

def main():
    """
    Hauptfunktion
    """
    # Pfade konfigurieren
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(script_dir, "mushroom_data.xlsx")
    output_dir = os.path.join(script_dir, "..", "..", "data", "scraped_mushrooms")
    
    # Prüfe ob Excel-Datei existiert
    if not os.path.exists(excel_file):
        logger.error(f"[FEHLER] Excel-Datei nicht gefunden: {excel_file}")
        return
    
    # Erstelle und starte Scraper
    scraper = MushroomImageScraper(excel_file, output_dir)
    
    try:
        scraper.run()
        logger.info("*** Mushroom Image Scraper erfolgreich abgeschlossen! ***")
        
    except KeyboardInterrupt:
        logger.info("*** Scraper durch Benutzer gestoppt ***")
    except Exception as e:
        logger.error(f"*** Scraper fehlgeschlagen: {e} ***")

if __name__ == "__main__":
    main()
