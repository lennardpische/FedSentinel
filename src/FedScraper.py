import requests
from bs4 import BeautifulSoup
import os
import time
import re

#Setup
BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SAVE_DIR = "data/raw_html"
os.makedirs(SAVE_DIR, exist_ok=True)

def scrape_statement_text(url):
    """
    Fetches article text for a specific statement link (HTML)
    """
    try: 
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to open {url}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')

        content = soup.find('div', id = 'article')

        if not content:
            content = soup.find('div', class_ = 'col-xs-12 col-md-8')
        if content:
            return content.get_text(separator='\n', strip = True)
    
    except Exception as e:
        print(f'Error scraping {url}: {e}')
        return None

def run_scraper():
    """
    Runs scraper algorithm
    """
    print(f'Accessing Calendar: {CALENDAR_URL}')
    response = requests.get(CALENDAR_URL)
    if response.status_code != 200:
        print(f"Could not access {url}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', href=True)
    count = 0

    for link in links:
        href = link['href']
        text = link.text.strip()
        if "monetary" in href and "pressreleases" in href and href.endswith(".htm"):
            if href.startswith('/'):
                full_url = BASE_URL + href
            elif href.startswith('http'):
                    full_url = href
            else:
                    full_url = f"{BASE_URL}/{href}"

            date_match = re.search(r'(\d{8})', full_url)

            if date_match:
                date_str = date_match.group(1)

                year = int(date_str[:4])
                if year <2000 or year >2026:
                    continue

                filename = f"{date_str}_Statement.txt"
                save_path = os.path.join(SAVE_DIR,filename)

            #unsure whether to keep; FOMC statements could be updated dynamically but wtv
                if os.path.exists(save_path):
                    print(f"Skipping {filename}. Already exists.")
                    continue

                print(f"Scraping {filename}...")
                text_content = scrape_statement_text(full_url)

                if text_content:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    count+= 1
                    time.sleep(1)
    print(f"Finished Scraping. Scraped {count} new statements.")

if __name__ == "__main__":
    run_scraper()


            
                

