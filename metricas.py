import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# Clase auxiliar idéntica para no romper la compatibilidad de lectura
class DatasetMapeado(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset.dataset[self.subset.indices[index]]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

if __name__ == "__main__":
    print("Cargando entorno para extracción de métricas oficiales...")

    CARPETA_DATASET = "dataset_comida_grande"
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Archivo del modelo que ya tienes entrenado (ajusta el nombre si es necesario)
    ARCHIVO_MODELO = "mejor_modelo_foodto.pth" 

    # Transformaciones idénticas a las de validación (sin aumentar datos artificialmente)
    transformaciones_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_completo = datasets.ImageFolder(CARPETA_DATASET, transform=None)
    class_names = dataset_completo.classes
    etiquetas = dataset_completo.targets

    # Reconstruimos exactamente el mismo split del 20% gracias al random_state=42
    indices = list(range(len(dataset_completo)))
    _, idx_val = train_test_split(
        indices, 
        test_size=0.20, 
        stratify=etiquetas, 
        random_state=42
    )

    split_val = Subset(dataset_completo, idx_val)
    val_dataset = DatasetMapeado(split_val, transform=transformaciones_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Reconstruimos la arquitectura exacta de tu ResNet18
    print("Cargando arquitectura ResNet18...")
    modelo = models.resnet18()
    num_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(256, len(class_names))
    )
    
    # Cargamos tus pesos ya entrenados sin tocar una sola neurona
    if os.path.exists(ARCHIVO_MODELO):
        modelo.load_state_dict(torch.load(ARCHIVO_MODELO, map_location=DISPOSITIVO))
        modelo = modelo.to(DISPOSITIVO)
        modelo.eval()
        print(f"¡Éxito! Pesos cargados desde '{ARCHIVO_MODELO}'")
    else:
        print(f"Error: No se encuentra el archivo {ARCHIVO_MODELO} en esta carpeta.")
        exit()

    # Extracción de predicciones
    todas_predicciones = []
    todas_etiquetas_reales = []
    
    print("Analizando el set de validación (Inferencia rápida)...")
    with torch.no_grad(): # Desactiva el cálculo de gradientes (no altera el modelo)
        for inputs, labels in val_loader:
            inputs = inputs.to(DISPOSITIVO)
            outputs = modelo(inputs)
            _, predicted = torch.max(outputs, 1)
            todas_predicciones.extend(predicted.cpu().numpy())
            todas_etiquetas_reales.extend(labels.numpy())

    # 1. Generar la Matriz de Confusión exigida en tu LaTeX
    print("Generando Matriz de Confusión...")
    cm = confusion_matrix(todas_etiquetas_reales, todas_predicciones)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - Foodto')
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción de la IA')
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=300)
    print("-> Archivo gráfico 'matriz_confusion.png' generado.")

    # 2. Calcular Exactitud, Precisión, Exhaustividad (Recall) y F1-Score en modo Macro Average
    print("Calculando reporte detallado por clases...")
    reporte = classification_report(
        todas_etiquetas_reales, 
        todas_predicciones, 
        labels=list(range(len(class_names))), 
        target_names=class_names
    )
    with open("metricas_finales_tfg.txt", "w", encoding="utf-8") as f:
        f.write("=== MÉTRICAS OBTENIDAS PARA LA MEMORIA DEL TFG ===\n\n")
        f.write(reporte)
        
    print("\n" + "="*50)
    print("¡PROCESO COMPLETADO EN SEGUNDOS!")
    print("="*50)
    print("Revisa el archivo 'metricas_finales_tfg.txt' para ver tus números.")
    print("La imagen 'matriz_confusion.png' está lista para tu documento LaTeX.")