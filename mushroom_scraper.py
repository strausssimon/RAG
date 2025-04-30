import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin
import sqlite3
from datetime import datetime

class MushroomScraper:
    def __init__(self, base_url="https://www.123pilze.de/DreamHC/Download/"):
        self.base_url = base_url
        self.db_path = "mushroom_images.db"
        self.excel_path = "mushroom_data.xlsx"
        self.setup_database()

    def setup_database(self):
        """Initialize SQLite database for storing images"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mushroom_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mushroom_name TEXT,
                image_data BLOB,
                image_filename TEXT,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()

    def scrape_mushroom_page(self, mushroom_name):
        """Scrape a specific mushroom page"""
        url = urljoin(self.base_url, f"{mushroom_name}.htm")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract mushroom information
            info = self.extract_mushroom_info(soup)
            
            # Extract and save images
            self.save_images(soup, mushroom_name)
            
            return info
            
        except requests.RequestException as e:
            print(f"Error scraping {mushroom_name}: {e}")
            return None

    def extract_mushroom_info(self, soup):
        """Extract text information about the mushroom"""
        info = {
            'name': '',
            'description': '',
            'habitat': '',
            'season': '',
            'edibility': ''
        }
        
        # Extract text content (specific selectors would need to be adjusted based on the website structure)
        # This is a template - you'll need to adjust the selectors based on the actual HTML structure
        main_content = soup.find('div', class_='content')
        if main_content:
            info['description'] = main_content.get_text(strip=True)
            
        return info

    def save_images(self, soup, mushroom_name):
        """Save images to SQLite database"""
        images = soup.find_all('img')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for img in images:
            try:
                img_url = urljoin(self.base_url, img.get('src', ''))
                if img_url:
                    img_response = requests.get(img_url)
                    img_response.raise_for_status()
                    
                    cursor.execute('''
                        INSERT INTO mushroom_images 
                        (mushroom_name, image_data, image_filename, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        mushroom_name,
                        img_response.content,
                        img_url.split('/')[-1],
                        datetime.now()
                    ))
            except requests.RequestException as e:
                print(f"Error downloading image {img_url}: {e}")

        conn.commit()
        conn.close()

    def save_to_excel(self, mushroom_data):
        """Save mushroom information to Excel file"""
        df = pd.DataFrame(mushroom_data)
        df.to_excel(self.excel_path, index=False)

    def scrape_multiple_mushrooms(self, mushroom_list):
        """Scrape multiple mushroom pages"""
        mushroom_data = []
        for mushroom in mushroom_list:
            info = self.scrape_mushroom_page(mushroom)
            if info:
                info['name'] = mushroom
                mushroom_data.append(info)
        
        # Save all data to Excel
        self.save_to_excel(mushroom_data)
