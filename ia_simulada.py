import json
import time

def consultar_alergenos_mock(plato):
    time.sleep(1)
    plato = plato.lower()
    
    respuesta = {}
    
    if "hamburger" in plato:
        respuesta = {
            "plato": "Hamburguesa Completa",
            "alergenos": ["Gluten", "Sésamo", "Mostaza", "Leche"],
            "riesgo": "ALTO",
            "info": "El pan suele llevar sésamo y leche. La salsa casi siempre mostaza."
        }
    elif "pizza" in plato:
        respuesta = {
            "plato": "Pizza Clásica",
            "alergenos": ["Gluten", "Leche"],
            "riesgo": "MEDIO",
            "info": "La masa es trigo puro. El queso es lácteo. Cuidado con el orégano (trazas)."
        }
    elif "sushi" in plato:
        respuesta = {
            "plato": "Sushi / Sashimi",
            "alergenos": ["Pescado", "Soja", "Gluten"],
            "riesgo": "ALTO",
            "info": "La salsa de soja lleva trigo. El surimi lleva huevo y crustáceos."
        }
    elif "pasta" in plato:
        respuesta = {
            "plato": "Pasta con Salsa",
            "alergenos": ["Gluten", "Huevo", "Leche"],
            "riesgo": "MEDIO",
            "info": "La pasta fresca suele llevar huevo. El parmesano es leche cruda."
        }
    elif "steak" in plato:
        respuesta = {
            "plato": "Filete de Carne",
            "alergenos": ["Ninguno (teóricamente)"],
            "riesgo": "BAJO",
            "info": "Plato seguro salvo contaminación en parrilla o salsas añadidas."
        }
    elif "salad" in plato:
        respuesta = {
            "plato": "Ensalada Mixta",
            "alergenos": ["Sulfitos (vinagre)", "Frutos secos"],
            "riesgo": "BAJO",
            "info": "Cuidado con los aliños industriales y los picatostes (Gluten)."
        }
    elif "fried rice" in plato:
        respuesta = {
            "plato": "Arroz Frito / Tres Delicias",
            "alergenos": ["Crustáceos", "Huevo", "Soja"],
            "riesgo": "ALTO",
            "info": "Las gambas son alérgenos potentes. Se cocina en wok con soja."
        }
    elif "cake" in plato:
        respuesta = {
            "plato": "Tarta / Pastel",
            "alergenos": ["Gluten", "Huevo", "Leche", "Frutos de cáscara"],
            "riesgo": "MUY ALTO",
            "info": "Bomba de alérgenos. Preguntar siempre por almendras o nueces."
        }
    elif "ice cream" in plato:
        respuesta = {
            "plato": "Helado",
            "alergenos": ["Leche", "Huevo", "Cacahuetes"],
            "riesgo": "MEDIO",
            "info": "Riesgo de trazas si usan la misma cuchara para varios sabores."
        }
    elif "sandwich" in plato:
        respuesta = {
            "plato": "Sándwich",
            "alergenos": ["Gluten", "Leche", "Soja"],
            "riesgo": "MEDIO",
            "info": "El pan de molde industrial suele llevar leche y soja."
        }
    else:
        respuesta = {
            "plato": plato.capitalize(),
            "alergenos": ["Desconocido"],
            "riesgo": "DESCONOCIDO",
            "info": "No se ha podido identificar con seguridad."
        }

    return respuesta

if __name__ == "__main__":
    print(json.dumps(consultar_alergenos_mock("cake"), indent=4))