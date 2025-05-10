import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import sqlite3
from datetime import datetime

class MushroomScraper:
    """A scraper for mushroom information from 123pilzsuche.de"""
    
    def __init__(self, base_url="https://www.123pilzsuche.de/daten/details/"):
        """Initialize the scraper with base URL and setup database"""
        self.base_url = base_url
        self.db_path = "mushroom_images.db"
        self.excel_path = "mushroom_data.xlsx"  # Fixed filename
        self.setup_database()

    def setup_database(self):
        """Initialize SQLite database for storing mushroom images"""
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

    def validate_mushroom_page(self, soup, expected_name):
        """Validate if the page contains the expected mushroom name"""
        # Try multiple locations for the mushroom name
        title_locations = [
            soup.find('span', style=lambda x: x and 'font-size:14.0pt' in x),
            soup.find('strong'),
            soup.find('h1'),
            soup.find('title')
        ]
        
        for loc in title_locations:
            if loc and loc.get_text(strip=True):
                text = loc.get_text(strip=True)
                if expected_name.lower() in text.lower():
                    return True
        return False

    def scrape_mushroom_page(self, mushroom_name):
        """Scrape a specific mushroom page"""
        url = urljoin(self.base_url, f"{mushroom_name}.htm")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # First validate and extract info
            info = self.extract_mushroom_info(soup)
            info['name'] = mushroom_name  # Ensure name is set
            return info if any(v for k, v in info.items() if k != 'name') else None
            
        except requests.RequestException:
            return None

    def extract_mushroom_info(self, soup):
        """Extract text information about the mushroom"""
        info = {
            'name': '',
            'taste': '',
            'smell': '',  # New field for Geruch
            'genus': ''   # New field for Gattung
        }
        
        # Find the main table - try multiple approaches
        tables = soup.find_all('table')
        main_table = None
        
        for table in tables:
            if table.find('td', string=lambda x: x and any(term in x.lower() for term in ['geschmack', 'geruch', 'gattung'])):
                main_table = table
                break
        
        if main_table:
            rows = main_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True).lower()
                    content = cells[1].get_text(strip=True)
                    
                    if 'geschmack' in header:
                        info['taste'] = content
                    elif 'geruch' in header:
                        info['smell'] = content
                    elif 'gattung' in header:
                        info['genus'] = content

        return info

    def save_images(self, soup, mushroom_name):
        """Save images to SQLite database"""
        # Look for images within the main content area
        main_table = soup.find('table', {'class': 'details'})
        if not main_table:
            return

        images = main_table.find_all('img')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for img in images:
            try:
                # Get the source URL of the image
                img_url = img.get('src', '')
                if not img_url:
                    continue
                    
                # Convert relative URL to absolute URL if necessary
                if not img_url.startswith('http'):
                    img_url = urljoin('https://www.123pilzsuche.de', img_url)
                
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
            except requests.RequestException:
                continue

        conn.commit()
        conn.close()

    def save_to_excel(self, mushroom_data):
        """Save mushroom information to Excel file with nice formatting"""
        # Create DataFrame with selected columns
        df = pd.DataFrame(mushroom_data)
        
        # Reorder and rename columns for better presentation
        columns = {
            'name': 'Pilzname',
            'taste': 'Geschmack',
            'smell': 'Geruch',
            'genus': 'Gattung'
        }
        
        df = df.rename(columns=columns)
        df = df[columns.values()]  # Reorder columns
        
        # Create Excel writer object with xlsxwriter engine
        with pd.ExcelWriter(self.excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Pilzdaten', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Pilzdaten']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'border': 1,
                'text_wrap': True
            })
            
            # Apply formats
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                # Set column width based on maximum length of data in each column
                max_length = max(
                    df[value].astype(str).apply(len).max(),
                    len(value)
                )
                worksheet.set_column(col_num, col_num, max_length + 2, cell_format)

    def scrape_multiple_mushrooms(self, mushroom_list):
        """Scrape multiple mushroom pages and save to Excel"""
        mushroom_data = []
        for mushroom in mushroom_list:
            info = self.scrape_mushroom_page(mushroom)
            if info:
                mushroom_data.append(info)
        
        if mushroom_data:
            self.save_to_excel(mushroom_data)
        return bool(mushroom_data)
