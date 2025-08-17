import requests
from bs4 import BeautifulSoup
from mushroom_scraper import MushroomScraper
from termcolor import colored
import sys

def test_mushroom_page(mushroom_name, expected_title):
    """Test scraping of a specific mushroom page with title validation"""
    scraper = MushroomScraper()
    
    # Get the raw HTML first for debugging
    url = scraper.base_url + f"{mushroom_name}.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the specific span with font-size:14.0pt that contains the title
    title_span = soup.find('span', style=lambda x: x and 'font-size:14.0pt' in x)
    
    # Find the taste information
    taste_cell = soup.find('span', string="Geschmack?")
    taste_value = None
    if taste_cell:
        # Navigate up to the table row and find the next cell
        row = taste_cell.find_parent('tr')
        if row:
            # Find all cells in the row and get the second one (index 1)
            cells = row.find_all('td')
            if len(cells) > 1:
                taste_value = cells[1].get_text(strip=True)
    
    if title_span:
        actual_title = title_span.get_text(strip=True).split('=')[0].strip()
        print(colored(f"🔍 Found title: '{actual_title}'", 'cyan'))
        
        if expected_title in actual_title:
            print(colored(f"✅ Test passed: '{expected_title}' successfully validated", 'green', attrs=['bold']))
            if taste_value:
                print(colored(f"🍄 Geschmack: {taste_value}", 'magenta'))
            else:
                print(colored("ℹ️ No taste information found", 'yellow'))
            return True
        else:
            print(colored(f"❌ Test failed: Expected '{expected_title}' but found '{actual_title}'", 'red', attrs=['bold']))
            return False
    else:
        print(colored("❌ Test failed: Title element not found in HTML", 'red', attrs=['bold']))
        return False

def run_tests():
    """Run all mushroom page tests"""
    test_cases = [
        ("Fliegenpilz", "Fliegenpilz"),
        ("AbweichenderSchueppling", "Abweichender Schüppling")
    ]
    
    results = []
    passed = 0
    failed = 0
    
    print(colored("\n🍄 === Mushroom Scraper Test Suite === 🍄", 'cyan', attrs=['bold']))
    print(colored("=" * 50, 'cyan'))
    
    for mushroom_name, expected_title in test_cases:
        print(colored(f"\n📋 Testing: {mushroom_name}", 'yellow', attrs=['bold']))
        print(colored("-" * 40, 'cyan'))
        result = test_mushroom_page(mushroom_name, expected_title)
        results.append((mushroom_name, result))
        if result:
            passed += 1
        else:
            failed += 1
    
    # Print final summary
    total = passed + failed
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(colored("\n📊 === Final Results === 📊", 'cyan', attrs=['bold']))
    print(colored("=" * 50, 'cyan'))
    print(f"Total Tests: {total}")
    print(colored(f"✅ Passed: {passed}", 'green'))
    print(colored(f"❌ Failed: {failed}", 'red'))
    print(colored(f"📈 Success Rate: {pass_rate:.1f}%", 'yellow', attrs=['bold']))
    
    if failed > 0:
        print(colored("\n❌ Some tests failed! Check the details above.", 'red', attrs=['bold']))
        sys.exit(1)
    else:
        print(colored("\n🎉 All tests passed successfully! 🎉", 'green', attrs=['bold']))
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
