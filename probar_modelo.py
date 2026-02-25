import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

RUTA_IMAGEN = "prueba.jpg"
RUTA_MODELO = "mi_modelo_comida.pth"

CLASES = [
    'cake',
    'fried rice',
    'hamburger',
    'ice cream',
    'pasta',
    'pizza',
    'salad',
    'sandwich',
    'steak',
    'sushi'
]

print(f"Analizando: {RUTA_IMAGEN}")

if not os.path.exists(RUTA_IMAGEN):
    print(f"Archivo no encontrado: {RUTA_IMAGEN}")
    exit()

try:
    imagen = Image.open(RUTA_IMAGEN).convert('RGB')
except Exception as e:
    print(f"Error: {e}")
    exit()

transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

imagen_tensor = transformacion(imagen).unsqueeze(0)

print(f"Cargando modelo...")
modelo = models.resnet18(weights=None) 
num_features = modelo.fc.in_features

modelo.fc = nn.Linear(num_features, len(CLASES)) 

try:
    modelo.load_state_dict(torch.load(RUTA_MODELO))
except RuntimeError as e:
    print("Error de tamaño:")
    print(f"{e}")
    exit()

modelo.eval()

with torch.no_grad():
    salida = modelo(imagen_tensor)
    porcentajes = torch.nn.functional.softmax(salida, dim=1)[0] * 100
    _, indice_ganador = torch.max(salida, 1)
    
    nombre_ganador = CLASES[indice_ganador]
    confianza = porcentajes[indice_ganador].item()

print("\nResultados:")
print("=" * 30)
print(f"Plato: {nombre_ganador.upper()}")
print(f"Confianza: {confianza:.2f}%")
print("=" * 30)

print("\nTop 3:")
probs_top3, indices_top3 = torch.topk(porcentajes, 3)
for i in range(3):
    idx = indices_top3[i].item()
    print(f"{i+1}. {CLASES[idx]}: {probs_top3[i]:.2f}%")