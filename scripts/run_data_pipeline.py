#!/usr/bin/env python3
"""
run_data_pipeline.py

A simple pipeline runner to orchestrate all data steps for the AI Menu Guide project:
- Scrape images
- Filter images with GPT
- Generate image embeddings

Usage:
  python run_data_pipeline.py [--step STEP] [--all]

Options:
  --step STEP   Run a specific step (scrape, filter, embed)
  --all         Run all steps in order (default: includes validate before embedding)

Edit script to adjust file paths or script arguments as needed.
"""
import subprocess
import argparse
import sys

# Paths to scripts
SCRAPER = 'dish_image_scraper.py'
FILTER = 'filter_scraped_images.py'
EMBED = 'image_embedding_generator.py'


def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Step failed: {desc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the full data pipeline or individual steps.")
    parser.add_argument('--step', choices=['scrape', 'filter', 'embed'], help='Run a specific step')
    parser.add_argument('--all', action='store_true', help='Run all steps in order')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument("--device", type=str, default='cpu', help="Device to use for computation (e.g., 'cuda', 'mps', 'cpu')")
    args = parser.parse_args()

    scraper_cmd = f'python {SCRAPER}'
    filter_cmd = f'python {FILTER}'
    embeder_cmd = f'python {EMBED}'
    if args.api_key:
        filter_cmd += f' --api-key {args.api_key}'
    if args.device:
        embeder_cmd += f' --device {args.device}'

    steps = [
        ('scrape', scraper_cmd, 'Scrape images'),
        ('filter', filter_cmd, 'Filter images with GPT'),
        ('embed', embeder_cmd, 'Generate image embeddings'),
    ]

    if args.all:
        for key, cmd, desc in steps:
            run_step(cmd, desc)
    elif args.step:
        for key, cmd, desc in steps:
            if key == args.step:
                run_step(cmd, desc)
                break
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
