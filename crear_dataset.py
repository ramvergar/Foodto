import os
import requests
from datasets import load_dataset
from PIL import Image
from io import BytesIO


CLASES_OBJETIVO = [
    "pizza", "hamburger", "sushi", "pasta", "steak", "salad", "cake", 
    "ice cream", "sandwich", "hot dog", "omelette", "fried chicken", 
    "tacos", "paella", "lasagna", "croissant", "donut", "guacamole",
    "prawns", "salmon", "fried rice", "spaghetti bolognese", "ramen",
    "fish and chips", "mousaka", "risotto", "cheesecake", "brownie",
    "pancakes", "waffles"
]


IMAGENES_POR_CLASE = 800 
CARPETA_DATASET = "dataset_comida_grande"

if not os.path.exists(CARPETA_DATASET):
    os.makedirs(CARPETA_DATASET)


contadores = {}
for clase in CLASES_OBJETIVO:
    ruta = os.path.join(CARPETA_DATASET, clase)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        contadores[clase] = 0
    else:
        
        archivos_existentes = [f for f in os.listdir(ruta) if f.endswith('.jpg')]
        contadores[clase] = len(archivos_existentes)

print("\n--- ESTADO ACTUAL DEL DATASET ---")
for clase, cantidad in contadores.items():
    if cantidad > 0:
        print(f"{clase.upper()}: {cantidad}/{IMAGENES_POR_CLASE} fotos ya descargadas.")
print("---------------------------------\n")

print(f"Conectando con Hugging Face. Objetivo: {len(CLASES_OBJETIVO)} clases.")

dataset = load_dataset("Codatta/MM-Food-100K", split="train", streaming=True)

total_objetivo = len(CLASES_OBJETIVO) * IMAGENES_POR_CLASE
print(f"Objetivo total: {total_objetivo} imágenes.")

for data in dataset:
    
    if sum(contadores.values()) >= total_objetivo:
        break

    if data['dish_name'] is None:
        continue
        
    nombre_plato = data['dish_name'].lower()
    
    
    clase_encontrada = None
    for clase in CLASES_OBJETIVO:
        if clase in nombre_plato: 
            if contadores[clase] < IMAGENES_POR_CLASE:
                clase_encontrada = clase
                break
    
    if clase_encontrada:
        url = data['image_url']
        try:
            
            response = requests.get(url, timeout=2) 
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                
                 
                
                nombre_fichero = f"{clase_encontrada}_{contadores[clase_encontrada]}.jpg"
                ruta_guardado = os.path.join(CARPETA_DATASET, clase_encontrada, nombre_fichero)
                
                img.save(ruta_guardado, "JPEG", quality=85)
                contadores[clase_encontrada] += 1
                
                if contadores[clase_encontrada] % 50 == 0:
                    print(f">>> {clase_encontrada.upper()}: {contadores[clase_encontrada]}/{IMAGENES_POR_CLASE}")
        except:
            continue 

print("\nDataset completado.")