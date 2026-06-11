import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

torch.backends.cudnn.benchmark = True

# Clase auxiliar indispensable para separar las transformaciones de Train y Validation
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

    print("Iniciando proceso de entrenamiento")

    CARPETA_DATASET = "dataset_comida_grande"
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {DISPOSITIVO}")

    # 1. SEPARACIÓN DE TRANSFORMACIONES
    transformaciones_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Robustez ante cambios de luz en fotos
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformaciones_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carga del dataset en crudo (sin transformaciones iniciales)
    dataset_completo = datasets.ImageFolder(CARPETA_DATASET, transform=None)
    class_names = dataset_completo.classes
    etiquetas = dataset_completo.targets

    print("Calculando pesos de clase para compensar el desbalanceo...")
    pesos_calculados = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(etiquetas),
        y=etiquetas
    )
    pesos_tensor = torch.tensor(pesos_calculados, dtype=torch.float).to(DISPOSITIVO)

    # 2. SEPARACIÓN ESTRATIFICADA PERFECTA (80% / 20% por cada plato de comida)
    indices = list(range(len(dataset_completo)))
    
    # Reparte equitativamente garantizando que todas las clases estén en train y val
    idx_train, idx_val = train_test_split(
        indices, 
        test_size=0.20, 
        stratify=etiquetas, # Fuerza a respetar las proporciones de cada comida
        random_state=42
    )

    # Creamos los subsets indexados
    split_train = Subset(dataset_completo, idx_train)
    split_val = Subset(dataset_completo, idx_val)

    # Asignamos las transformaciones correctas a cada split de forma independiente
    train_dataset = DatasetMapeado(split_train, transform=transformaciones_train)
    val_dataset = DatasetMapeado(split_val, transform=transformaciones_val)

    # DATALOADERS (Manteniendo tus optimizaciones de velocidad de lectura)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True, # Shuffle=True garantiza mezcla total de platos en cada nueva época
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Numero de clases: {len(class_names)}")
    print(f"Distribucion de datos: {len(train_dataset)} entrenamiento, {len(val_dataset)} validacion")

    # 3. CONFIGURACIÓN CON FINE-TUNING PARCIAL Y CLASIFICADOR ROBUSTO
    print("Configurando arquitectura ResNet18...")
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Congelamos toda la base primero
    for param in modelo.parameters():
        param.requires_grad = False  

    # DESCONGELAMOS la última capa convolucional (layer4) para especializarla en texturas de platos
    for param in modelo.layer4.parameters():
        param.requires_grad = True

    # Rediseñamos el clasificador final (Mucha más capacidad que la capa lineal única anterior)
    num_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4), # Protege al modelo de memorizar el dataset de entrenamiento
        nn.Linear(256, len(class_names))
    )
    modelo = modelo.to(DISPOSITIVO)

    criterio = nn.CrossEntropyLoss(weight=pesos_tensor)

    # El optimizador solo actualizará las capas libres (layer4 y la nueva fc)
    optimizador = optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=0.0001)

    num_epochs = 50
    mejor_precision_val = 0.0
    epocas_sin_mejorar = 0
    limite_paciencia = 6

    print(f"Inicio de entrenamiento: {num_epochs} epocas maximas")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):

        modelo.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DISPOSITIVO, non_blocking=True)
            labels = labels.to(DISPOSITIVO, non_blocking=True)
            optimizador.zero_grad()
            
            outputs = modelo(inputs)
            loss = criterio(outputs, labels)
            loss.backward()
            optimizador.step()
            
            running_loss += loss.item()
            
        loss_promedio = running_loss / len(train_loader)

        modelo.eval()
        val_correctos = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DISPOSITIVO), labels.to(DISPOSITIVO)
                outputs = modelo(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correctos += (predicted == labels).sum().item()

        val_acc = 100 * val_correctos / val_total
        
        print(f"Epoca {epoch+1:02d}/{num_epochs} | Loss: {loss_promedio:.4f} | Accuracy Val: {val_acc:.2f}%")

        if val_acc > mejor_precision_val:
            mejor_precision_val = val_acc
            epocas_sin_mejorar = 0
            torch.save(modelo.state_dict(), "mejor_modelo_foodto.pth")
            print("Actualizacion de mejor modelo realizada.")
        else:
            epocas_sin_mejorar += 1
            print(f"Paciencia de Early Stopping: {epocas_sin_mejorar}/{limite_paciencia}")
            
            if epocas_sin_mejorar >= limite_paciencia:
                print("Entrenamiento finalizado por criterio de Early Stopping.")
                break

    time_elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Tiempo total de ejecucion: {time_elapsed/60:.2f} minutos")
    print(f"Precision maxima alcanzada: {mejor_precision_val:.2f}%")