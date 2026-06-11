import os
import torch
import torch.nn as nn
from django.apps import AppConfig
from torchvision import models


class FoodtoConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Foodto"

    modelo_ia = None

    # !!! ATENCIÓN: Esta lista DEBE tener las 26 clases de tu dataset en ORDEN ALFABÉTICO !!!
    # Ejemplo aproximado (añade aquí las que te falten hasta tener las 26 exactas de "dataset_comida_grande"):
    clases_ia = [
        "cake",
        "cheesecake",
        "croissant",
        "donut",
        "fried chicken",
        "fried rice",
        "hamburger",
        "hot dog",
        "ice cream",
        "lasagna",
        "omelette",
        "paella",
        "pancakes",
        "pasta",
        "pizza",
        "prawns",
        "ramen",
        "risotto",
        "salad",
        "salmon",
        "sandwich",
        "spaghetti bolognese",
        "steak",
        "sushi",
        "tacos",
        "waffles",
    ]  

    def ready(self):
        """Esta función se ejecuta UNA SOLA VEZ cuando arranca el servidor Django."""

        if os.environ.get("RUN_MAIN") == "true":
            print("[Foodto] Despertando a la IA de reconocimiento visual...")

            # 1. Instanciamos la base de ResNet18 vacía
            modelo = models.resnet18(weights=None)
            num_features = modelo.fc.in_features

            # 2. CALCAMOS EXACTAMENTE la misma estructura que usamos para entrenar el modelo del 82%
            modelo.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),  # En modo .eval() esto se desactiva solo, no te preocupes
                nn.Linear(256, len(self.clases_ia)),
            )

            # 3. Buscamos el archivo de pesos
            ruta_modelo = os.path.join(
                os.path.dirname(__file__), "mi_modelo_comida.pth"
            )

            if os.path.exists(ruta_modelo):
                # Cargamos los pesos forzando el dispositivo CPU (ideal para servidores web locales)
                modelo.load_state_dict(
                    torch.load(ruta_modelo, map_location=torch.device("cpu"), weights_only=True)
                )
                modelo.eval()  # Congela las capas de Dropout y BatchNormalization para inferencia

                self.modelo_ia = modelo
                print(
                    f"[Foodto] IA del 82% cargada con éxito. Detectando {len(self.clases_ia)} clases."
                )
            else:
                print(
                    f"[Foodto] ERROR: No encuentro el modelo entrenado en {ruta_modelo}"
                )