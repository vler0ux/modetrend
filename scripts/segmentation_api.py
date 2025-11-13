"""
Script de segmentation d'images de vêtements via l'API Hugging Face
Modèle utilisé : segformer_b3_clothes
"""

import os
import requests
import time
from dotenv import load_dotenv
from PIL import Image
import io
import json
from typing import Union, Dict, Any
import base64

# Charger les variables d'environnement
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Configuration de l'API
API_URL = "https://api-inference.huggingface.co/models/mattmdjaga/segformer_b3_clothes"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Taille maximale recommandée pour l'API (en pixels)
MAX_IMAGE_SIZE = 1024


def optimize_image(image_path: str, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    """
    Optimise une image pour l'envoi à l'API.
    
    Args:
        image_path: Chemin vers l'image
        max_size: Taille maximale (largeur ou hauteur) en pixels
    
    Returns:
        bytes: Image optimisée en bytes
    """
    try:
        # Charger l'image
        img = Image.open(image_path)
        
        # Convertir en RGB si nécessaire
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionner si nécessaire
        width, height = img.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  Image redimensionnée de {width}x{height} à {new_size[0]}x{new_size[1]}")
        
        # Convertir en bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    except Exception as e:
        raise Exception(f"Erreur lors de l'optimisation de l'image : {e}")


def query_segmentation(image_path: str, max_retries: int = 3, retry_delay: int = 5) -> Dict[Any, Any]:
    """
    Envoie une image au modèle de segmentation via l'API Hugging Face.
    
    Args:
        image_path: Chemin vers l'image à segmenter
        max_retries: Nombre maximum de tentatives en cas d'erreur
        retry_delay: Délai entre les tentatives (secondes)
    
    Returns:
        dict: Résultats de la segmentation
    """
    print(f"\n{'='*60}")
    print(f"Traitement de : {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    # Optimiser l'image
    print("Optimisation de l'image...")
    image_bytes = optimize_image(image_path)
    image_size_mb = len(image_bytes) / (1024 * 1024)
    print(f"  Taille de l'image : {image_size_mb:.2f} MB")
    
    # Effectuer la requête avec gestion des erreurs
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nTentative {attempt}/{max_retries}...")
            print("Envoi de la requête à l'API...")
            
            response = requests.post(
                API_URL,
                headers=HEADERS,
                data=image_bytes,
                timeout=60  # Timeout de 60 secondes
            )
            
            # Vérifier le statut de la réponse
            if response.status_code == 200:
                print("✓ Réponse reçue avec succès !")
                result = response.json()
                return result
            
            elif response.status_code == 503:
                # Modèle en cours de chargement
                print("⏳ Le modèle est en cours de chargement...")
                try:
                    error_data = response.json()
                    estimated_time = error_data.get('estimated_time', retry_delay)
                    print(f"   Temps d'attente estimé : {estimated_time}s")
                    time.sleep(estimated_time + 2)
                except:
                    time.sleep(retry_delay)
                continue
            
            elif response.status_code == 429:
                # Limite de taux atteinte
                print("⚠️ Limite de requêtes atteinte")
                print(f"   Attente de {retry_delay}s avant nouvelle tentative...")
                time.sleep(retry_delay)
                continue
            
            elif response.status_code == 401:
                raise Exception("Erreur d'authentification. Vérifiez votre token HF_TOKEN.")
            
            else:
                print(f"✗ Erreur HTTP {response.status_code}")
                print(f"   Message : {response.text}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Échec après {max_retries} tentatives")
        
        except requests.exceptions.Timeout:
            print(f"✗ Timeout de la requête (> 60s)")
            if attempt < max_retries:
                print(f"   Nouvelle tentative dans {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise Exception("Timeout : le serveur ne répond pas")
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Erreur de connexion : {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise


def display_segmentation_results(result: Dict[Any, Any], image_path: str) -> None:
    """
    Affiche les résultats de la segmentation de manière lisible.
    
    Args:
        result: Résultats retournés par l'API
        image_path: Chemin de l'image originale
    """
    print(f"\n{'='*60}")
    print("RÉSULTATS DE LA SEGMENTATION")
    print(f"{'='*60}")
    
    if isinstance(result, list) and len(result) > 0:
        print(f"\nNombre de segments détectés : {len(result)}")
        print("\nDétails des segments :\n")
        
        for i, segment in enumerate(result, 1):
            label = segment.get('label', 'Inconnu')
            score = segment.get('score', 0)
            
            print(f"  {i}. {label}")
            print(f"     Confiance : {score:.2%}")
            
            # Informations supplémentaires si disponibles
            if 'mask' in segment:
                print(f"     Masque : Disponible")
    else:
        print("Aucun segment détecté ou format de résultat inattendu")
        print(f"Résultat brut : {result}")


def save_results(result: Dict[Any, Any], image_path: str, output_dir: str = "output") -> str:
    """
    Sauvegarde les résultats de la segmentation dans un fichier JSON.
    
    Args:
        result: Résultats de la segmentation
        image_path: Chemin de l'image originale
        output_dir: Répertoire de sortie
    
    Returns:
        str: Chemin du fichier de résultats
    """
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Nom du fichier de sortie
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_segmentation.json")
    
    # Sauvegarder les résultats
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Résultats sauvegardés dans : {output_path}")
    return output_path


def main():
    """
    Fonction principale pour tester le script.
    """
    # Vérifier que le token est configuré
    if not HF_TOKEN:
        print("✗ Erreur : Token HF_TOKEN non trouvé dans .env")
        return
    
    print("✓ Token Hugging Face chargé")
    
    # Exemple : traiter une image
    # Remplacez par le chemin de votre image
    image_path = "data/images/IMG/image_35.png"
    
    if not os.path.exists(image_path):
        print(f"\n⚠️ Image d'exemple non trouvée : {image_path}")
        print("Veuillez placer une image dans data/images/ et modifier le chemin dans le script")
        return
    
    try:
        # Effectuer la segmentation
        result = query_segmentation(image_path)
        
        # Afficher les résultats
        display_segmentation_results(result, image_path)
        
        # Sauvegarder les résultats
        save_results(result, image_path)
        
        print(f"\n{'='*60}")
        print("✓ Traitement terminé avec succès !")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n✗ Erreur : {e}")


if __name__ == "__main__":
    main()