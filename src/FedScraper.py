import requests
from bs4 import BeautifulSoup
import os
import time

#Setup
BASE_URL = "https://www.federalreserve.gov"
YEARS = range(2020, 2026) # post-Covid years
SAVE_DIR = "data/raw_html"
os.makedirs("SAVE_DIR", exist_ok=True)

def get_FOMC_links(year):
    """
    Finds all 'Statement' links for a given year
    """
    url = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []