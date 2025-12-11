"""
This is for data collection for the nj2025_270toWin.csv dataset.
Step -1:
Scrape New Jersey 2025 Governor Polling Data from 270toWin.com
- Parse the HTML table containing polling data
- Extract relevant fields: pollster, dates, sample size, percentages, margin
- Save to CSV for further processing

Some other design choices:
We have decided to use a functional method, where each python script is a function that can be imported and called from other scripts.
This allows for modularity and reusability across different parts of the project.
"""
import re
from datetime import datetime 
from urllib.parse import urljoin

import requests
import pandas as pd 
from bs4 import BeautifulSoup

URL = "https://www.270towin.com/2025-governor-polls/new-jersey"
HEADERS = {"User-Agent": "Mozilla/5.0 (NJYouthProject/1.0)"}

def parse_dates_270(text: str):
   
    text = text.strip()
    try:
        dt = datetime.strptime(text, "%m/%d/%Y").date()
        return dt, dt
    except Exception:
        return None, None


def parse_sample(text: str):
    if not text:
        return None, None

    cleaned = (
        text.replace("±", "")
            .replace("&plusmn", "")
            .replace("–", "-")
            .strip()
    )

    m = re.search(r"([\d,]+)\s*(LV|RV|A)\b", cleaned, re.I)
    if not m:
        return None, None

    size_str = m.group(1).replace(",", "")
    pop = m.group(2).upper()

    try:
        size = int(size_str)
    except ValueError:
        return None, None

    return size, pop


def parse_pct(s: str):
    s = (s or "").replace("%", "").strip()
    if not s:
        return None
    return float(s)

#main 
def main():
    print("Fetching 270toWin page...")
    resp = requests.get(URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if not table:
        print("Could not find a <table> on the page.")
        return

    rows = table.find_all("tr")
    if len(rows) < 2:
        print("Table has no data rows.")
        return

    header_cells = rows[0].find_all(["th", "td"])
    headers = [c.get_text(strip=True) for c in header_cells]
    print("Detected headers:", headers)

    records = []

    for r in rows[1:]:
        tds = r.find_all("td")
        if not tds:
            continue

        cols = [td.get_text(strip=True) for td in tds]

        if len(cols) >= 7 and cols[0] == "":
            cols = cols[1:]

        if len(cols) < 6:
            continue

        pollster = cols[0]

        if "average of" in pollster.lower():
            continue

        date_text = cols[1]
        sample_text = cols[2]
        sherrill_raw = cols[3]
        ciatt_raw = cols[4]

        #parse date
        start_date, end_date = parse_dates_270(date_text)
        if start_date is None:
            continue


        sample_size, population = parse_sample(sample_text)
        if sample_size is None:
            continue

     
        sherrill = parse_pct(sherrill_raw)
        ciatt = parse_pct(ciatt_raw)
        if sherrill is None or ciatt is None:
            continue

        margin = round(sherrill - ciatt, 2)

        poll_url = None
        for a in r.find_all("a", href=True):
            href = a["href"]
            full = urljoin(URL, href)
            if (
                href.lower().endswith(".pdf")
                or "doc_upload" in href.lower()
                or "poll" in href.lower()
            ):
                poll_url = full
                break

        records.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "pollster": pollster,
                "sample_size": sample_size,
                "population": population,
                "sherril_pct": sherrill,   # Sherrill
                "opponent_pct": ciatt,      # Ciattarelli
                "margin": margin,
                "youthvote_agerange": "N/A",
                "youthvote_s": "N/A",
                "youthvote_c": "N/A",
                "poll_url": poll_url,
            }
        )

    if not records:
        print("No polls parsed.")
        return

    df = pd.DataFrame(records)
    # Sort newest first
    df.sort_values("start_date", ascending=False, inplace=True)

    out_file = "nj2025_270toWinOG.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file} with {len(df)} polls.")


if __name__ == "__main__":
    main()
