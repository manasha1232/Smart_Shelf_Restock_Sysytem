"""
Demo Shelf Image Generator
==========================
Generates a realistic 3-tier composite shelf test image (input/shelf.jpg)
containing real objects (cups, bowls, books, oranges) on Top, Middle, 
and Bottom shelves to demonstrate the Smart Shelf Restock Alert System.
"""

import os
import urllib.request
import cv2
import numpy as np


def generate_demo_shelf(output_path="input/shelf.jpg"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download sample open-source photographic assets if not present
    assets = {
        "books": ("input/_asset_books.jpg", "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=800"),
        "items": ("input/_asset_items.jpg", "https://images.unsplash.com/photo-1563227812-0ea4c22e6cc8?w=800")
    }

    for key, (local_file, url) in assets.items():
        if not os.path.exists(local_file):
            print(f"[DEMO] Fetching photographic asset '{key}'...")
            try:
                urllib.request.urlretrieve(url, local_file)
            except Exception as e:
                print(f"[WARN] Could not fetch asset {key}: {e}")

    books_img = cv2.imread("input/_asset_books.jpg")
    items_img = cv2.imread("input/_asset_items.jpg")

    if books_img is None or items_img is None:
        print("[WARN] Using fallback canvas generation...")
        canvas = np.ones((600, 600, 3), dtype=np.uint8) * 220
        cv2.imwrite(output_path, canvas)
        return

    # 600x600 Shelf Canvas
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 230

    # Draw Wooden Horizontal Shelf Bars
    wood_color = (60, 100, 150)
    border_color = (30, 50, 80)
    
    # Top Shelf Bar (y=190-205)
    cv2.rectangle(canvas, (0, 190), (600, 205), wood_color, -1)
    cv2.rectangle(canvas, (0, 190), (600, 205), border_color, 2)

    # Middle Shelf Bar (y=390-405)
    cv2.rectangle(canvas, (0, 390), (600, 405), wood_color, -1)
    cv2.rectangle(canvas, (0, 390), (600, 405), border_color, 2)

    # Bottom Shelf Bar (y=585-600)
    cv2.rectangle(canvas, (0, 585), (600, 600), wood_color, -1)
    cv2.rectangle(canvas, (0, 585), (600, 600), border_color, 2)

    # Paste item crops onto shelves
    # Top Shelf (Shelf 1: y=0-190) -> Items (Cups, Orange, Bowl)
    patch_top = cv2.resize(items_img[50:350, 100:400], (300, 175))
    canvas[15:190, 150:450] = patch_top

    # Middle Shelf (Shelf 2: y=205-390) -> Books
    patch_mid = cv2.resize(books_img[50:350, 100:500], (360, 180))
    canvas[210:390, 120:480] = patch_mid

    # Bottom Shelf (Shelf 3: y=405-585) -> Items (Cups, Bowl)
    patch_bot = cv2.resize(items_img[100:400, 200:500], (320, 175))
    canvas[410:585, 140:460] = patch_bot

    cv2.imwrite(output_path, canvas)
    print(f"[DEMO] Successfully created realistic multi-shelf test image at: {output_path}")


if __name__ == "__main__":
    generate_demo_shelf()
