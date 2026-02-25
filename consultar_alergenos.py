import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") 

# Configurar la conexión con Google Gemini
genai.configure(api_key=API_KEY)

def obtener_info_nutricional(nombre_plato):
    print(f"Preguntando a la IA sobre: {nombre_plato}...")
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Actúa como un experto tecnólogo de alimentos y nutricionista.
    Analiza el siguiente plato: "{nombre_plato}".
    
    Tu tarea es identificar los alérgenos obligatorios según el Reglamento UE 1169/2011.
    
    Devuelve la respuesta ÚNICAMENTE en formato JSON con esta estructura exacta:
    {{
        "plato": "{nombre_plato}",
        "ingredientes_clave": ["lista", "de", "ingredientes", "comunes"],
        "alergenos_seguros": ["lista", "de", "alergenos", "muy", "probables"],
        "posibles_trazas": ["lista", "de", "trazas", "por", "contaminacion"],
        "advertencia": "Un breve consejo de seguridad de una frase."
    }}
    
    No añadas texto antes ni después del JSON.
    """
    
    try:
        response = model.generate_content(prompt)
        
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        
        datos = json.loads(texto_limpio)
        return datos

    except Exception as e:
        return {"error": f"Error: {e}"}

if __name__ == "__main__":
    plato_detectado = "hamburger"
    
    resultado = obtener_info_nutricional(plato_detectado)
    
    print("\n--- 🥗 REPORTE DE ALÉRGENOS (JSON) ---")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))