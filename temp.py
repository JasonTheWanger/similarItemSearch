import requests
from bs4 import BeautifulSoup

def fetch_google_doc_table_data(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table_rows = soup.find_all('tr')

    data = []
    for row in table_rows:
        cells = row.find_all('td')
        if len(cells) != 3:
            continue
        try:
            x = int(cells[0].get_text(strip=True))
            char = cells[1].get_text(strip=True)
            y = int(cells[2].get_text(strip=True))
            data.append((char, x, y))
        except:
            continue

    return data

def render_grid(data):
    if not data:
        print("No character data found.")
        return

    max_x = max(x for _, x, _ in data)
    max_y = max(y for _, _, y in data)
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for char, x, y in data:
        grid[y][x] = char

    for row in grid:
        print(''.join(row))


if __name__ == "__main__":
    url = "https://docs.google.com/document/d/e/2PACX-1vTER-wL5E8YC9pxDx43gk8eIds59GtUUk4nJo_ZWagbnrH0NFvMXIw6VWFLpf5tWTZIT9P9oLIoFJ6A/pub"
    data = fetch_google_doc_table_data(url)
    render_grid(data)
