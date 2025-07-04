#!/usr/bin/env python3
"""
Image Embedding Generator
Generates CLIP embeddings for dish images using Xenova/clip-vit-base-patch32 model.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle
from tqdm import tqdm
import argparse
import sys


class ImageEmbeddingGenerator:
    def __init__(self, model_name="Xenova/clip-vit-base-patch32", device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def preprocess_image(self, image_path):
        """Preprocess image for CLIP model"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def generate_embedding(self, image_path):
        """Generate embedding for a single image"""
        try:
            inputs = self.preprocess_image(image_path)
            if inputs is None:
                return None
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                embedding = image_features.cpu().numpy().flatten()
                
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return None
    
    def process_image_directory(self, image_dir, output_file="embeddings.json", scraped_json=None):
        """Process all images in a directory and generate embeddings.
         If scraped_json is provided, use it to add URLs to metadata."""
        print(f"Processing images from: {image_dir}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        print(f"Found {len(image_files)} images")
        url_lookup = {}
        if scraped_json and os.path.exists(scraped_json):
            with open(scraped_json, 'r', encoding='utf-8') as f:
                for rec in json.load(f):
                    url_lookup[os.path.abspath(rec['filename'])] = rec.get('url', '')
        embeddings_data = []
        for image_path in tqdm(image_files, desc="Generating embeddings"):
            embedding = self.generate_embedding(image_path)
            if embedding is not None:
                relative_path = os.path.relpath(image_path, image_dir)
                cuisine = relative_path.split(os.sep)[0] if os.sep in relative_path else "unknown"
                filename = os.path.basename(image_path)
                dish_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                url = url_lookup.get(os.path.abspath(image_path), "")
                embeddings_data.append({
                    "relative_path": relative_path,
                    "cuisine": cuisine,
                    "dish": dish_name,
                    "url": url,
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding)
                })
        self.save_embeddings(embeddings_data, output_file)
        print(f"Generated embeddings for {len(embeddings_data)} images")
        print(f"Embeddings saved to: {output_file}")
        return embeddings_data
    
    def process_scraped_images(self, scraped_images_file, output_file="dish_embeddings.json"):
        """Process images from scraped images JSON file"""
        print(f"Processing scraped images from: {scraped_images_file}")

        with open(scraped_images_file, 'r') as f:
            scraped_data = json.load(f)
        
        print(f"Found {len(scraped_data)} scraped images")

        embeddings_data = []
        
        for item in tqdm(scraped_data, desc="Generating embeddings"):
            image_path = item["filename"]

            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue

            embedding = self.generate_embedding(image_path)
            
            if embedding is not None:
                embeddings_data.append({
                    "cuisine": item["cuisine"],
                    "dish": item["dish"],
                    "url": item.get("url", ""),
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding)
                })

        self.save_embeddings(embeddings_data, output_file)
        
        print(f"Generated embeddings for {len(embeddings_data)} images")
        print(f"Embeddings saved to: {output_file}")
        
        return embeddings_data
    
    def save_embeddings(self, embeddings_data, output_file):
        """Save embeddings to JSON file"""

        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)

        # Save as JS file for fast loading in the web app
        js_file = output_file.replace('.json', '.js')
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write('const dishEmbeddings = ')
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            f.write(';\n')
        
        print(f"Embeddings saved as JSON: {output_file}")
        print(f"Embeddings saved as JS: {js_file}")
    
    def load_embeddings(self, embeddings_file):
        """Load embeddings from file"""
        if embeddings_file.endswith('.json'):
            with open(embeddings_file, 'r') as f:
                return json.load(f)
        elif embeddings_file.endswith('.pkl'):
            with open(embeddings_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")
    
    def find_similar_images(self, query_image_path, embeddings_data, top_k=5):
        """Find similar images using cosine similarity"""

        query_embedding = self.generate_embedding(query_image_path)
        
        if query_embedding is None:
            print("Could not generate embedding for query image")
            return []

        similarities = []
        for item in embeddings_data:
            item_embedding = np.array(item["embedding"])
            similarity = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            similarities.append((similarity, item))

        similarities.sort(key=lambda x: x[0], reverse=True)

        return similarities[:top_k]
    
    def search_by_text(self, text_query, embeddings_data, top_k=5):
        """Search images by text query"""

        inputs = self.processor(text=text_query, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            text_embedding = text_features.cpu().numpy().flatten()

        similarities = []
        for item in embeddings_data:
            item_embedding = np.array(item["embedding"])
            similarity = np.dot(text_embedding, item_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(item_embedding)
            )
            similarities.append((similarity, item))

        similarities.sort(key=lambda x: x[0], reverse=True)

        return similarities[:top_k]


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Image Embedding Generator")
    parser.add_argument("--input", type=str, default='dish_images/filtered_progress.json', help="Path to the input JSON file with image metadata (from scraper) or directory")
    parser.add_argument("--output", type=str, default="dish_images/dish_embeddings.json", help="Path to save the output JSON file with embeddings")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Name of the CLIP model to use")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use for computation (e.g., 'cuda', 'mps', 'cpu')")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--scraped-json", type=str, default=None, help="Optional: path to scraping_progress.json for URL lookup")
    args = parser.parse_args()
    print("=== Image Embedding Generator ===")
    print(f"Loading CLIP model: {args.model}")
    generator = ImageEmbeddingGenerator(model_name=args.model, device=args.device)
    try:
        if os.path.isdir(args.input):
            generator.process_image_directory(args.input, args.output, scraped_json=args.scraped_json)
        elif args.input.endswith('.json'):
            generator.process_scraped_images(args.input, args.output)
        else:
            print(f"Input must be a directory or JSON file: {args.input}")
            return
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise
    sys.exit(0)


if __name__ == "__main__":
    main()
