import os
import json
import base64
from io import BytesIO
import torch
from torchvision import transforms
from PIL import Image

from django.shortcuts import render, redirect
from django.apps import apps
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login

from Foodto.models import AnalisisComida, HistorialAnalisis
from groq import Groq  


client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def obtener_analisis_clinico_llama(nombre_plato):
    prompt_sistema = (
        "Eres un sistema experto en seguridad alimentaria, alergias y nutrición clínica. "
        "Analiza el plato que te indique el usuario y responde ÚNICAMENTE con un objeto JSON válido en español. "
        "No añadas textos de introducción, ni saludos, ni conclusiones. Sé directo. "
        "La estructura del JSON debe ser estrictamente la siguiente:\n"
        "{\n"
        '  "riesgo": "BAJO" o "MEDIO" o "ALTO" o "MUY ALTO",\n'
        '  "alergenos": ["Nombre Alérgeno 1", "Nombre Alérgeno 2"],\n'
        '  "info": "Escribe un párrafo serio e informativo indicando qué personas intolerantes, '
        'o en qué situaciones de salud específicas (como mujeres embarazadas, personas con hipertensión, '
        'problemas cardíacos, diabetes u otros tipos de enfermedades) deben evitar este alimento o tener precauciones especiales al consumirlo."\n'
        "}" 
    )

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Plato: {nombre_plato}"}
            ],
            response_format={"type": "json_object"},  
            temperature=0.2  
        )
        return json.loads(chat_completion.choices[0].message.content)

    except Exception as e:
        print(f"\nERROR EN LA LLAMADA A GROQ: {e}\n")
        
        return {
            "riesgo": "MEDIO",
            "alergenos": ["Clasificación temporal no disponible"],
            "info": f"No se pudo conectar de forma dinámica con el módulo de análisis clínico (Llama API). Por precaución, consulte los ingredientes con el establecimiento."
        }


def inicio_view(request):
    if request.user.is_authenticated:
        return redirect('escaner')
    return render(request, 'Foodto/inicio.html')


def registro(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            usuario = form.save() 
            auth_login(request, usuario)
            return redirect('escaner') 
    else:
        form = UserCreationForm()
    return render(request, 'Foodto/registro.html', {'form': form})


@login_required
def perfil(request):
    historial = HistorialAnalisis.objects.filter(usuario=request.user).order_by('-fecha')
    contexto = {
        'historial': historial,
        'total_analisis': historial.count()
    }
    return render(request, 'Foodto/perfil.html', contexto)


@login_required
def escaner_view(request):
    contexto = {} 
    
    if request.method == 'POST' and request.FILES.get('foto'):
        foto_subida = request.FILES['foto']
        
        # 1. LEEMOS EN RAM
        foto_bytes = foto_subida.read()
        
        # 2. CONVERTIMOS A BASE64 PARA EL HTML
        imagen_b64 = base64.b64encode(foto_bytes).decode('utf-8')
        contexto['url_imagen'] = f"data:image/jpeg;base64,{imagen_b64}"
        
        # 3. PROCESAMIENTO DE IMAGEN CON PYTORCH
        imagen = Image.open(BytesIO(foto_bytes)).convert('RGB')
        transformacion = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        imagen_tensor = transformacion(imagen).unsqueeze(0)
        
        config_app = apps.get_app_config('Foodto')
        modelo = config_app.modelo_ia
        clases = config_app.clases_ia
        
        if modelo is None:
            contexto['error'] = "Error grave: El cerebro de la IA no se cargó al encender el servidor."
            return render(request, 'Foodto/index.html', contexto)
            
        with torch.no_grad():
            salida = modelo(imagen_tensor.cpu())
            porcentajes = torch.nn.functional.softmax(salida, dim=1)[0] * 100
            _, indice_ganador = torch.max(salida, 1)

            plato_detectado = clases[indice_ganador].lower()
            confianza = porcentajes[indice_ganador].item()
            

        plato_formateado = plato_detectado.capitalize()
        datos_nutricionales = obtener_analisis_clinico_llama(plato_formateado)
        
        # Guardamos datos en el contexto respetando las variables exactas de tu HTML
        contexto['resultado'] = plato_formateado
        contexto['plato'] = plato_formateado 
        contexto['precision'] = round(confianza, 2)
        contexto['confianza'] = round(confianza, 2)
        
        # Inyección de los datos estructurados generados por Groq
        contexto['alergenos'] = datos_nutricionales.get("alergenos", [])
        contexto['riesgo'] = datos_nutricionales.get("riesgo", "MEDIO").upper()
        contexto['info'] = datos_nutricionales.get("info", "")
        
        # Guardado en el historial de usuario
        if request.user.is_authenticated: 
            HistorialAnalisis.objects.create(
                usuario=request.user,
                plato=plato_formateado
            )

    return render(request, 'Foodto/index.html', contexto)