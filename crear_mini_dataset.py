import os
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO

CLASES_OBJETIVO = [
    "pizza", "hamburger", "sushi", "pasta", "steak", 
    "salad", "fried rice", "cake", "ice cream", "sandwich"
]

IMAGENES_POR_CLASE = 50
CARPETA_DATASET = "dataset_comida"

if not os.path.exists(CARPETA_DATASET):
    os.makedirs(CARPETA_DATASET)

for clase in CLASES_OBJETIVO:
    ruta = os.path.join(CARPETA_DATASET, clase)
    if not os.path.exists(ruta):
        os.makedirs(ruta)

print(f"Conectando con Hugging Face para buscar {len(CLASES_OBJETIVO)} tipos de comida...")
dataset = load_dataset("Codatta/MM-Food-100K", split="train", streaming=True)

print(f"Objetivo: {IMAGENES_POR_CLASE} fotos por cada una de las 10 clases.")

contadores = {clase: 0 for clase in CLASES_OBJETIVO}
total_objetivo = len(CLASES_OBJETIVO) * IMAGENES_POR_CLASE

for data in dataset:
    if data['dish_name'] is None:
        continue
        
    if sum(contadores.values()) >= total_objetivo:
        break

    nombre_plato = data['dish_name'].lower()
    
    clase_encontrada = None
    for clase in CLASES_OBJETIVO:
        if clase in nombre_plato: 
            clase_encontrada = clase
            break
    
    if clase_encontrada and contadores[clase_encontrada] < IMAGENES_POR_CLASE:
        url = data['image_url']
        if url:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    
                    nombre_fichero = f"{clase_encontrada}_{contadores[clase_encontrada]}.jpg"
                    ruta_guardado = os.path.join(CARPETA_DATASET, clase_encontrada, nombre_fichero)
                    
                    img.save(ruta_guardado)
                    contadores[clase_encontrada] += 1
                    
                    print(f"[{clase_encontrada.upper()}] Foto guardada ({contadores[clase_encontrada]}/{IMAGENES_POR_CLASE})")
            except:
                pass 

print("Dataset completado")