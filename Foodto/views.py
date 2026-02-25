from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.apps import apps
from ia_simulada import consultar_alergenos_mock
import torch
from torchvision import transforms
from PIL import Image

def escaner_view(request):
    contexto = {} 
    
    
    if request.method == 'POST' and request.FILES.get('foto'):
        foto_subida = request.FILES['foto']
        
        fs = FileSystemStorage()
        nombre_archivo = fs.save(foto_subida.name, foto_subida)
        url_foto = fs.url(nombre_archivo)
        contexto['url_imagen'] = url_foto
        
        try:
            imagen = Image.open(fs.path(nombre_archivo)).convert('RGB')
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
                
            datos_nutricionales = consultar_alergenos_mock(plato_detectado)
            
            contexto['plato'] = plato_detectado.capitalize()
            contexto['confianza'] = round(confianza, 2)
            contexto['alergenos'] = datos_nutricionales.get("alergenos", [])
            contexto['riesgo'] = datos_nutricionales.get("riesgo", "DESCONOCIDO")
            contexto['info'] = datos_nutricionales.get("info", "")
            
        except Exception as e:
            contexto['error'] = f"Error procesando la imagen: {str(e)}"

    return render(request, 'Foodto/index.html', contexto)