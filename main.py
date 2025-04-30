from mushroom_scraper import MushroomScraper

def main():
    # Initialize the scraper
    scraper = MushroomScraper()
    
    # List of mushrooms to scrape
    mushrooms_to_scrape = [
        'Fliegenpilz',
        'Fruehlingsegerling'
        # Add more mushrooms here
    ]
    
    # Scrape all mushrooms
    scraper.scrape_multiple_mushrooms(mushrooms_to_scrape)

if __name__ == "__main__":
    main()
