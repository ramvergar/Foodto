import os
import sys

print("Diagnostico iniciado")

print(f"Python {sys.version}")

carpeta = "dataset_comida"
if os.path.exists(carpeta):
    print(f"La carpeta '{carpeta}' existe.")
    pizzas = os.path.join(carpeta, "pizza")
    if os.path.exists(pizzas):
        num_fotos = len(os.listdir(pizzas))
        print(f"Hay {num_fotos} fotos en pizza.")
        if num_fotos == 0:
            print("Carpeta vacia.")
    else:
        print("Carpeta pizza no existe.")
else:
    print(f"Carpeta {carpeta} no existe.")

print("Importando PyTorch...")
try:
    import torch
    print(f"PyTorch {torch.__version__}")
    
    from torchvision import datasets, transforms
    print("Torchvision importado.")
    
    try:
        transformaciones = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        datos = datasets.ImageFolder(carpeta, transform=transformaciones)
        print(f"Dataset cargado. {len(datos)} fotos totales.")
        print(f"Clases: {datos.classes}")
    except Exception as e:
        print(f"Error al cargar dataset: {e}")

except ImportError:
    print("PyTorch no instalado.")
except Exception as e:
    print(f"Error: {e}")

print("Diagnostico completado")