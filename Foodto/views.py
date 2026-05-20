from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.apps import apps
from Foodto.models import AnalisisComida, HistorialAnalisis
from ia_simulada import consultar_alergenos_mock
import torch
from torchvision import transforms
from PIL import Image
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
import base64
from io import BytesIO

def inicio_view(request):
    # Si el usuario ya ha iniciado sesión, lo mandamos directo al escáner
    if request.user.is_authenticated:
        return redirect('escaner')
    
    # Si no, le mostramos la pantalla de botones
    return render(request, 'Foodto/inicio.html')

def registro(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login') # Te manda al login tras registrarte
    else:
        form = UserCreationForm()
    return render(request, 'Foodto/registro.html', {'form': form})

# Vista de Perfil (Solo accesible si estás logueado)
@login_required
def perfil(request):
    # Buscamos todos los platos de este usuario, ordenados del más nuevo al más antiguo
    historial = HistorialAnalisis.objects.filter(usuario=request.user).order_by('-fecha')
    
    contexto = {
        'historial': historial,
        'total_analisis': historial.count()
    }
    return render(request, 'Foodto/perfil.html', contexto)


def escaner_view(request):
    contexto = {} 
    
    if request.method == 'POST' and request.FILES.get('foto'):
        foto_subida = request.FILES['foto']
        
        # 1. LEEMOS EN RAM (Adiós a las fotos guardadas en el PC)
        foto_bytes = foto_subida.read()
        
        # 2. CONVERTIMOS A BASE64 PARA EL HTML
        imagen_b64 = base64.b64encode(foto_bytes).decode('utf-8')
        contexto['url_imagen'] = f"data:image/jpeg;base64,{imagen_b64}"
        
        # 3. SIN TRY/EXCEPT: Si falla, queremos ver la pantalla amarilla de Django
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
            salida = modelo(imagen_tensor)
            porcentajes = torch.nn.functional.softmax(salida, dim=1)[0] * 100
            _, indice_ganador = torch.max(salida, 1)
            
            plato_detectado = clases[indice_ganador].lower()
            confianza = porcentajes[indice_ganador].item()
            
        # Ojo: si esta función no existe en tu archivo, aquí petará (y lo veremos)
        datos_nutricionales = consultar_alergenos_mock(plato_detectado)
        
        # Le mandamos los nombres por duplicado para que el HTML acierte seguro
        contexto['resultado'] = plato_detectado.capitalize()
        contexto['plato'] = plato_detectado.capitalize() 
        
        contexto['precision'] = round(confianza, 2)
        contexto['confianza'] = round(confianza, 2)
        
        contexto['alergenos'] = datos_nutricionales.get("alergenos", [])
        contexto['riesgo'] = datos_nutricionales.get("riesgo", "DESCONOCIDO")
        contexto['info'] = datos_nutricionales.get("info", "")
        
        # Ojo: si no has importado AnalisisComida, aquí petará (y lo veremos)
        if request.user.is_authenticated: 
            HistorialAnalisis.objects.create(
                usuario=request.user,
                plato=plato_detectado.capitalize()
            )

    return render(request, 'Foodto/index.html', contexto)