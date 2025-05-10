from mushroom_scraper import MushroomScraper

def main():
    scraper = MushroomScraper()
    mushrooms = [
        "Fliegenpilz",
        "AbweichenderSchueppling"
    ]
    
    if scraper.scrape_multiple_mushrooms(mushrooms):
        print(f"Daten wurden in {scraper.excel_path} gespeichert.")
    else:
        print("Keine Daten gefunden.")

if __name__ == "__main__":
    main()
