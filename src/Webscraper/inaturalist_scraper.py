"""
====================================================
Programmname : Mushroom Image Scraper
Datum        : 17.08.2025
Version      : 1.0
Beschreibung : Scraper für Pilzbilder von iNaturalist 

====================================================
"""
import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Konfiguration
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_mushrooms.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_boletus_edulis.csv.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_Tylopilus_felleus.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_amanita_phalloides.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_armillaria_mellea.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_Imleria_badia.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_Suillellus_luridus.csv" #Agaricus silvaticus
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_agaricus_silvaticus.csv"
# OLD: CSV_FILE = "data/csv/inaturalist/inaturalist_cantharellus_cibarius.csv"
CSV_FILE = "data/csv/inaturalist/inaturalist_phallus_impudicus.csv"

# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Amanita_muscaria"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Boletus_edulis"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Tylopilus_felleus"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Amanita_phalloides"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Armillaria_mellea"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/Imleria_badia"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/agaricus_silvaticus"
# OLD: OUTPUT_DIR = "data/inaturalist_mushrooms/cantharellus_cibarius"
OUTPUT_DIR = "data/inaturalist_mushrooms/phallus_impudicus"

# OLD: FIXED_NAME = "amanita_muscaria"
# OLD: FIXED_NAME = "boletus_edulis"
# OLD: FIXED_NAME = "tylopilus_felleus"
# OLD: FIXED_NAME = "amanita_phalloides"
# OLD: FIXED_NAME = "armillaria_mellea"
# OLD: FIXED_NAME = "Imleria_badia"
# OLD: FIXED_NAME = "agaricus_silvaticus"
# OLD: FIXED_NAME = "cantharellus_cibarius"
FIXED_NAME = "phallus_impudicus"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_id_from_url(image_url):
    # Extrahiere die ID aus der URL, z.B. .../837/medium.jpg -> 837
    parts = image_url.split('/')
    for i, part in enumerate(parts):
        if part.isdigit():
            return part
    # Fallback: nehme vorletzten Teil, falls keine reine Ziffer gefunden
    if len(parts) > 2:
        return parts[-2]
    return "unknown"

def get_extension(image_url):
    ext = os.path.splitext(urlparse(image_url).path)[1]
    return ext if ext else ".jpg"

def main():
    if not os.path.exists(CSV_FILE):
        print(f"CSV-Datei nicht gefunden: {CSV_FILE}")
        return
    df = pd.read_csv(CSV_FILE)
    if 'image_url' not in df.columns:
        print("CSV muss 'image_url' Spalte enthalten!")
        return
    for _, row in df.iterrows():
        image_url = row['image_url']
        img_id = get_id_from_url(image_url)
        ext = get_extension(image_url)
        filename = f"{FIXED_NAME}_{img_id}{ext}"
        save_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(save_path):
            print(f"Übersprungen (existiert): {filename}")
            continue
        try:
            resp = requests.get(image_url, timeout=20)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            print(f"Gespeichert: {filename}")
        except Exception as e:
            print(f"Fehler bei {image_url}: {e}")

if __name__ == "__main__":
    main()
