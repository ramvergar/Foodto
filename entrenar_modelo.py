import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

print("Iniciando entrenamiento")

CARPETA_DATASET = "dataset_comida"
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DISPOSITIVO}")

print("Cargando fotos...")

transformaciones_entrenamiento = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datos = datasets.ImageFolder(CARPETA_DATASET, transform=transformaciones_entrenamiento)
cargador = torch.utils.data.DataLoader(datos, batch_size=16, shuffle=True)

class_names = datos.classes
print(f"Clases ({len(class_names)}): {class_names}")
print(f"Total: {len(datos)} imagenes")

print("Cargando ResNet18...")
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in modelo.parameters():
    param.requires_grad = True 

num_features = modelo.fc.in_features
modelo.fc = nn.Linear(num_features, len(class_names))
modelo = modelo.to(DISPOSITIVO)

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.0001)

num_epochs = 15

print(f"Iniciando {num_epochs} epocas")
print("-" * 30)

start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    correctos = 0
    total = 0
    
    for inputs, labels in cargador:
        inputs, labels = inputs.to(DISPOSITIVO), labels.to(DISPOSITIVO)
        optimizador.zero_grad()
        outputs = modelo(inputs)
        loss = criterio(outputs, labels)
        loss.backward()
        optimizador.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correctos += (predicted == labels).sum().item()

    acc = 100 * correctos / total
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoca {epoch+1}/{num_epochs}: Error={running_loss:.4f} | Precision={acc:.2f}%")

time_elapsed = time.time() - start_time
print("-" * 30)
print(f"Entrenamiento completado en {time_elapsed/60:.1f} minutos")

torch.save(modelo.state_dict(), "mi_modelo_comida.pth")
print("Modelo guardado.")