# AI-Powered Menu Guide 🍽️

This project is a web application that uses AI to analyze handwritten menu images, extract dish names, translate them to English, and find visually similar dishes from a curated database. The goal is to help users understand and visualize unfamiliar dishes from menus in any language.

The project's live demo runs at https://jeniasivets.github.io/menu-guide/

## Overview
- Upload a photo of a handwritten menu (any language)
- The app uses OpenAI's GPT API to extract and translate dish names
- For each dish, the app finds the most visually similar dishes from a database using CLIP embeddings
- The top 3 similar dish images are shown, along with key ingredients

![Demo CountPages alpha](assets/demo.gif)

## Features
### Tech stack
- **Vanilla JavaScript + CSS**: Modern ES6+ features with async/await
- **Transformers.js**: Client-side CLIP model inference using Xenova/clip-vit-base-patch32
- **Real-time Processing**: Live console output and progress indicators


### Models
- **OpenAI GPT-4**: Analyse text on image + Multi-step validation for image quality filtering
- **CLIP (Contrastive Language-Image Pre-training)**: Visual similarity matching


### Notes
- You need an OpenAI API key to use the app (enter it in the web interface)
- All processing happens in the browser except for the initial data pipeline
- The app is designed to run on GitHub Pages or any static hosting


## Project Structure
- `app.js`, `index.html`, `style.css`: The web application
- `data/`: Embedding metadata for the app
- `scripts/`: All data pipeline scripts and dish lists to collect and process images



## Data Pipeline
The image database is built and maintained with a set of scripts that automate the collection, filtering, and embedding of dish images.

[//]: # (### Processing Pipeline)
- **Image Scraping**: Automated collection of high-quality dish images
- **Filtering**: GPT-powered image quality and relevance filtering
- **Embedding Generation**: CLIP embeddings for visual similarity search

Individual Steps

```bash
cd scripts

# scrape images
python dish_image_scraper.py

# filter out irrelevant or low-quality images
python filter_scraped_images.py --api-key

# computes CLIP embeddings for all images and stores metadata
python image_embedding_generator.py --device
```
or Complete Pipeline
```bash
python run_data_pipeline.py --all --device --api-key
```

## Local Development Setup


- To run data pipeline clone repo, install Python 3.10 and required packages:
   ```bash
   pip install torch transformers pillow requests beautifulsoup4 tqdm numpy
   cd scripts
   python run_data_pipeline.py --all --device --api-key
   ```
- To start a local server and use the web app:
    ```bash
    # Run development server
    python -m http.server 8000
    ```