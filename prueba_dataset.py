from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

print("Conectando al dataset...")

dataset = load_dataset("Codatta/MM-Food-100K", split="train", streaming=True)

iterator = iter(dataset)

print("Buscando imagen...")

found = False
count = 0

while not found and count < 10:
    try:
        data = next(iterator)
        url = data['image_url']
        
        if url:
            print(f"Descargando: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                print("\nExito. Plato encontrado.")
                print("-" * 40)
                print(f"Plato: {data['dish_name']}")
                print(f"Tipo: {data['food_type']}")
                print(f"Ingredientes: {data['ingredients']}")
                print("-" * 40)
                
                img.show()
                found = True
            else:
                print("Error descargando, probando siguiente...")
        count += 1
    except Exception as e:
        print(f"Error: {e}")
        count += 1

if not found:
    print("No se descargaron imagenes.")