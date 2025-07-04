#!/usr/bin/env python3
"""
Improved Dish Image Scraper (Configurable)
Reads hierarchical dish list from dish_lists.json, searches for high-quality images, and downloads them.
All configuration is via command-line arguments.
"""
import os
import json
import time
import requests
from PIL import Image
import io
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import re
import argparse
import numpy as np


def flatten_dish_list(dish_dict, path=None):
    if path is None:
        path = []
    flat = []
    for key, value in dish_dict.items():
        if isinstance(value, list):
            for dish in value:
                if isinstance(dish, dict) and "name" in dish:
                    flat.append((path[0] if path else key, key, dish))
        elif isinstance(value, dict):
            flat.extend(flatten_dish_list(value, path + [key]))
    return flat


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_large_image(img_bytes, min_width, min_height):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        return w >= min_width and h >= min_height
    except Exception:
        return False


def download_image(url, min_width, min_height, min_filesize, headers):
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('image'):
            if len(resp.content) >= min_filesize and is_large_image(resp.content, min_width, min_height):
                return resp.content
    except Exception:
        pass
    return None


def google_image_search(query, max_results=10, headers=None):
    search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        image_urls = []
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src and src.startswith('http') and not src.endswith('.gif'):
                image_urls.append(src)
            if len(image_urls) >= max_results:
                break
        return image_urls
    except Exception:
        return []


def bing_image_search(query, max_results=10, headers=None):
    search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}"
    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        image_urls = []
        for a in soup.find_all('a', class_='iusc'):
            m = re.search(r'"murl":"(https:[^\"]+)"', a.get('m', ''))
            if m:
                url = m.group(1).replace('\\u002f', '/')
                image_urls.append(url)
            if len(image_urls) >= max_results:
                break
        if not image_urls:
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src.startswith('http') and not src.endswith('.gif'):
                    image_urls.append(src)
                if len(image_urls) >= max_results:
                    break
        return image_urls
    except Exception:
        return []


def duckduckgo_image_search(query, max_results=10, headers=None):
    try:
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"
        resp = requests.get(url, headers=headers, timeout=10)
        vqd = re.search(r'vqd=([\d-]+)&', resp.text)
        if not vqd:
            vqd = re.search(r'vqd=([\d-]+)', resp.text)
        if not vqd:
            return []
        vqd = vqd.group(1)
        api_url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(query)}&vqd={vqd}"
        resp = requests.get(api_url, headers=headers, timeout=10)
        data = resp.json()
        return [r['image'] for r in data.get('results', [])[:max_results]]
    except Exception:
        return []


def is_blacklisted(url, blacklist_keywords):
    url_lower = url.lower()
    return any(keyword in url_lower for keyword in blacklist_keywords)


def is_reasonable_aspect_ratio(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        aspect = w / h if h != 0 else 0
        return 0.7 < aspect < 1.5
    except Exception:
        return False


def is_colorful(img_bytes, threshold=20):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        arr = np.array(img)
        (r, g, b) = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        colorfulness = np.sqrt(std_rg**2 + std_yb**2)
        return colorfulness > threshold
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser(description="Scrape dish images from the web using a dish list.")
    parser.add_argument('--output-dir', type=str, default='dish_images', help='Directory to save images and progress file')
    parser.add_argument('--max-images', type=int, default=10, help='Max images per dish')
    parser.add_argument('--progress-file', type=str, default=None, help='Path to save progress JSON (default: <output-dir>/scraping_progress.json)')
    parser.add_argument('--dish-list', type=str, default='../data/dish_lists.json', help='Path to dish_lists.json')
    parser.add_argument('--min-width', type=int, default=300, help='Minimum image width')
    parser.add_argument('--min-height', type=int, default=300, help='Minimum image height')
    parser.add_argument('--min-filesize', type=int, default=20*1024, help='Minimum image file size in bytes')
    parser.add_argument('--user-agent', type=str, default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', help='User-Agent header for requests')
    args = parser.parse_args()

    output_dir = args.output_dir
    max_images = args.max_images
    progress_file = args.progress_file or os.path.join(output_dir, 'scraping_progress.json')
    dish_list_file = args.dish_list
    min_width = args.min_width
    min_height = args.min_height
    min_filesize = args.min_filesize
    HEADERS = {'User-Agent': args.user_agent}
    BLACKLIST_KEYWORDS = [
        'store', 'logo', 'icon', 'vector', 'menu', 'sign', 'ad', 'stock', 'cartoon', 'drawing', 'clipart', 'illustration', 'banner', 'poster', 'symbol', 'button', 'emoji', 'emoticon', 'avatar', 'animation', 'sketch', 'painting', 'art', 'graphic', 'template', 'background', 'frame', 'label', 'price', 'discount', 'sale', 'offer', 'deal', 'shopping', 'buy', 'sell', 'order', 'delivery', 'restaurant', 'fastfood', 'cafe', 'bar', 'pub', 'drink', 'beverage', 'juice', 'soda', 'water', 'milk', 'coffee', 'tea', 'beer', 'wine', 'cocktail', 'liquor', 'alcohol', 'spirit', 'champagne', 'whiskey', 'vodka', 'rum', 'gin', 'brandy', 'cognac', 'liqueur', 'aperitif', 'digestif', 'mocktail', 'smoothie', 'shake', 'frappe', 'slush', 'ice', 'cream', 'dessert', 'cake', 'pie', 'tart', 'pudding', 'custard', 'mousse', 'souffle', 'brownie', 'cookie', 'biscuit', 'cracker', 'wafer', 'bar', 'candy', 'chocolate', 'sweet', 'sugar', 'honey', 'jam', 'jelly', 'marmalade', 'spread', 'butter', 'cheese', 'yogurt', 'curd', 'paneer', 'tofu', 'egg', 'omelette', 'scramble', 'boil', 'poach', 'fry', 'bake', 'roast', 'grill', 'barbecue', 'smoke', 'steam', 'stew', 'soup', 'broth', 'stock', 'consomme', 'bouillon', 'chowder', 'bisque', 'gazpacho', 'minestrone', 'goulash', 'borscht', 'tomato', 'vegetable', 'fruit', 'salad', 'greens', 'lettuce', 'spinach', 'kale', 'arugula', 'rocket', 'cabbage', 'coleslaw', 'slaw', 'kimchi', 'sauerkraut', 'pickle', 'relish', 'chutney', 'salsa', 'dip', 'spread', 'paste', 'puree', 'mash', 'mousse', 'foam', 'gel', 'jelly', 'pate', 'terrine', 'galantine', 'aspic', 'headcheese', 'brawn', 'souse', 'scrapple', 'liver', 'kidney', 'heart', 'tongue', 'tripe', 'sweetbread', 'brain', 'marrow', 'tail', 'feet', 'trotter', 'hock', 'shank', 'rib', 'loin', 'chop', 'cutlet', 'steak', 'roast', 'joint', 'rack', 'crown', 'saddle', 'haunch', 'leg', 'shoulder', 'breast', 'wing', 'drumstick', 'thigh', 'fillet', 'tenderloin', 'sirloin', 'rump', 'flank', 'brisket', 'plate', 'shortrib', 'back', 'neck', 'cheek', 'jowl', 'snout', 'ear', 'tail', 'hoof', 'horn', 'antler', 'bone', 'cartilage', 'gristle', 'fat', 'suet', 'lard', 'tallow', 'oil', 'butter', 'ghee', 'margarine', 'shortening', 'dripping', 'schmaltz', 'bacon', 'ham', 'sausage', 'salami', 'bologna', 'mortadella', 'prosciutto', 'pancetta', 'guanciale', 'lardo', 'coppa', 'capicola', 'soppressata', 'nduja', 'chorizo', 'linguica', 'andouille', 'boudin', 'blood', 'black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey', 'gray', 'beige', 'tan', 'ivory', 'cream', 'gold', 'silver', 'bronze', 'copper', 'pewter', 'platinum']

    ensure_dir(output_dir)
    with open(dish_list_file, 'r', encoding='utf-8') as f:
        dish_dict = json.load(f)
    flat_dishes = flatten_dish_list(dish_dict)
    downloaded_images = []

    for cuisine, subcuisine, dish_obj in flat_dishes:
        dish_name = dish_obj["name"]
        print(f"\nSearching for: {dish_name} ({cuisine}/{subcuisine})")
        cuisine_dir = os.path.join(output_dir, cuisine)
        ensure_dir(cuisine_dir)
        found = 0
        tried_urls = set()
        query_variants = [
            f"{dish_name} food dish",
            f"traditional {dish_name} food",
            f"homemade {dish_name}",
            f"{dish_name} recipe",
            f"{dish_name} cooked dish"
        ]
        for searcher in [google_image_search, bing_image_search, duckduckgo_image_search]:
            for query in query_variants:
                image_urls = searcher(query, max_results=15, headers=HEADERS)
                for url in image_urls:
                    if url in tried_urls or is_blacklisted(url, BLACKLIST_KEYWORDS):
                        continue
                    tried_urls.add(url)
                    img_bytes = download_image(url, min_width, min_height, min_filesize, HEADERS)
                    if img_bytes and is_reasonable_aspect_ratio(img_bytes) and is_colorful(img_bytes):
                        safe_dish_name = dish_name.replace(" ", "_").replace("/", "_")
                        filename = os.path.join(cuisine_dir, f"{safe_dish_name}_{found+1}.jpg")
                        with open(filename, 'wb') as out:
                            out.write(img_bytes)
                        downloaded_images.append({
                            "cuisine": cuisine,
                            "subcategory": subcuisine,
                            "dish": dish_name,
                            "filename": filename,
                            "url": url
                        })
                        print(f"  Downloaded: {filename}")
                        found += 1
                        if found >= max_images:
                            break
                    time.sleep(0.5)
                if found >= max_images:
                    break
            if found >= max_images:
                break
        if found == 0:
            print(f"  No suitable images found for {dish_name}")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(downloaded_images, f, indent=2)
    print(f"\n=== Scraping completed! ===")
    print(f"Total images downloaded: {len(downloaded_images)}")
    print(f"Results saved to: {progress_file}")


if __name__ == "__main__":
    main()
