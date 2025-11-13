"""
Script de test simple pour la segmentation
"""

import os
from dotenv import load_dotenv
from segmentation_api import query_segmentation, display_segmentation_results

# Charger le token
load_dotenv()

# Chemin vers votre image de test
image_path = "data/images/test.jpg"

if os.path.exists(image_path):
    print("Lancement de la segmentation...")
    result = query_segmentation(image_path)
    display_segmentation_results(result, image_path)
else:
    print(f"⚠️ Placez une image de test dans : {image_path}")