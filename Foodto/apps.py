from django.apps import AppConfig
import torch
import torch.nn as nn
from torchvision import models
import os

class FoodtoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Foodto' 

    #
    modelo_ia = None
    clases_ia = [
        'cake', 'fried rice', 'hamburger', 'ice cream', 'pasta', 
        'pizza', 'salad', 'sandwich', 'steak', 'sushi'
    ]

    def ready(self):
        """
        Esta función se ejecuta UNA SOLA VEZ cuando arranca el servidor Django.
        Es el lugar perfecto para cargar modelos pesados de Machine Learning.
        """
        
        if os.environ.get('RUN_MAIN') == 'true':
            print("[Foodto] Despertando a la IA de reconocimiento visual...")
            
            
            modelo = models.resnet18(weights=None)
            num_features = modelo.fc.in_features
            modelo.fc = nn.Linear(num_features, len(self.clases_ia))
            
            
            ruta_modelo = os.path.join(os.path.dirname(__file__), 'mi_modelo_comida.pth')
            
            if os.path.exists(ruta_modelo):
                
                modelo.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
                modelo.eval() 
                
                
                self.modelo_ia = modelo
                print("[Foodto] IA cargada en memoria y lista para escanear.")
            else:
                print(f"[Foodto] ERROR: No encuentro el modelo en {ruta_modelo}")