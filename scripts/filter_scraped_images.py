import os
import json
import requests
from tqdm import tqdm
import argparse


def ask_gpt_filter_image(image_url, dish_name, api_key, api_url, model):
    prompt = f"""
You are a food image expert. Given the following image and dish name, answer the following questions with only 'yes' or 'no' for each (in order, separated by commas):

1. Does this image represent the dish? (Is it a correct match for the dish name?)
2. Is this a real photo of a cooked food dish (not a drawing, logo, menu, or people)?
3. Does this image contain visible text or a watermark?
4. Does this image contain people?

Image URL: {image_url}
Dish name: {dish_name}

Return your answer as a comma-separated list of 'yes' or 'no' (e.g., 'yes,yes,no,no'). Do not add any explanation.
"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': model,
        'messages': [
            {"role": "system", "content": "You are a food image expert."},
            {"role": "user", "content": prompt}
        ],
        'max_tokens': 10,
        'temperature': 0.0,
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()['choices'][0]['message']['content'].strip().lower()
        parts = [x.strip() for x in answer.split(',')]
        if len(parts) == 4:
            is_dish, is_food_photo, has_text_or_watermark, has_people = parts
            return (
                is_dish.startswith('yes') and
                is_food_photo.startswith('yes') and
                has_text_or_watermark.startswith('no') and
                has_people.startswith('no')
            )
        else:
            print(f"Unexpected GPT answer format: {answer}")
            return False
    else:
        print(f"Error from GPT API: {response.status_code} {response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Filter scraped images using GPT for dish relevance and quality.")
    parser.add_argument('--input', type=str, default='dish_images/scraping_progress.json', help='Input JSON file with scraped images')
    parser.add_argument('--output', type=str, default='dish_images/filtered_progress.json', help='Output JSON file for filtered images')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--api-url', type=str, default='https://api.openai.com/v1/chat/completions', help='OpenAI API URL')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano', help='OpenAI model name')
    args = parser.parse_args()

    input_json = args.input
    output_json = args.output or input_json
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    api_url = args.api_url
    model = args.model

    if not api_key:
        print("Error: OpenAI API key must be provided via --api-key or OPENAI_API_KEY env var.")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        images = json.load(f)

    filtered = []
    for rec in tqdm(images, desc='Filtering images'):
        url = rec.get('url')
        dish = rec.get('dish')
        if not url or not dish:
            continue
        try:
            passed = ask_gpt_filter_image(url, dish, api_key, api_url, model)
        except Exception as e:
            print(f"Error for {url}: {e}")
            passed = False
        if passed:
            filtered.append(rec)
        else:
            print(f"Filtered out: {dish} | {url}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Filtered {len(images) - len(filtered)} images. Remaining: {len(filtered)}")


if __name__ == '__main__':
    main()
