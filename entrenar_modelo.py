import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
import os
import time
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    print("Iniciando proceso de entrenamiento")


    CARPETA_DATASET = "dataset_comida_grande"
    DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {DISPOSITIVO}")


    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carga del dataset original
    dataset_completo = datasets.ImageFolder(CARPETA_DATASET, transform=transformaciones)
    class_names = dataset_completo.classes
    etiquetas = dataset_completo.targets


    print("Calculando pesos de clase para compensar el desbalanceo...")
    pesos_calculados = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(etiquetas),
        y=etiquetas
    )
    pesos_tensor = torch.tensor(pesos_calculados, dtype=torch.float).to(DISPOSITIVO)


    total_fotos = len(dataset_completo)
    train_size = int(0.8 * total_fotos)
    val_size = total_fotos - train_size

    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Numero de clases: {len(class_names)}")
    print(f"Distribucion de datos: {train_size} entrenamiento, {val_size} validacion")


    print("Configurando arquitectura ResNet18...")
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in modelo.parameters():
        param.requires_grad = False  

    num_features = modelo.fc.in_features
    modelo.fc = nn.Linear(num_features, len(class_names))
    modelo = modelo.to(DISPOSITIVO)

    criterio = nn.CrossEntropyLoss(weight=pesos_tensor)

    optimizador = optim.Adam(modelo.fc.parameters(), lr=0.0001)


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