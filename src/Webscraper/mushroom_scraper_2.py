import pandas as pd
import requests
import os
from pathlib import Path

class MushroomImageScraper:
    def __init__(self, excel_path, image_dir):
        self.df = pd.read_excel(excel_path)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def download_images(self, mushroom_name=None):
        """Download images for specific mushroom or all if none specified"""
        if mushroom_name:
            to_download = self.df[self.df['name'] == mushroom_name]
        else:
            to_download = self.df
            
        for idx, row in to_download.iterrows():
            try:
                image_url = row['image']
                name = row['name']
                
                # Create filename from mushroom name and URL
                file_extension = os.path.splitext(image_url)[1]
                filename = f"{name.replace(' ', '_')}{file_extension}"
                filepath = self.image_dir / filename
                
                # Download image
                response = requests.get(image_url)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {image_url}")
                    
            except Exception as e:
                print(f"Error processing {row['name']}: {e}")

if __name__ == "__main__":
    excel_path = "/Users/celineotten/Documents/Git/SmallLanguageModels/SmallLanguageModels-1/smalllanguagemodels/Webscraper/data/mushroom_data.xlsx"
    image_dir = "../data/images"
    
    scraper = MushroomImageScraper(excel_path, image_dir)
    scraper.download_images()  # Download all images
    # Or download specific mushroom:
    # scraper.download_images("Radulomyces molaris")