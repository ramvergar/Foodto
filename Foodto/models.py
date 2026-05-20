from django.db import models
from django.contrib.auth.models import User  # Importamos el modelo de usuarios de Django

class AnalisisComida(models.Model):
    # Relacionamos el análisis con un usuario. Si el usuario se borra, se borra su historial.
    usuario = models.ForeignKey(User, on_delete=models.CASCADE, related_name='historial')
    
    # Datos que queremos guardar del análisis
    nombre_plato = models.CharField(max_length=100)
    porcentaje_acierto = models.FloatField()
    fecha_analisis = models.DateTimeField(auto_now_add=True) # Se guarda la hora automáticamente
    
    # Opcional: podrías guardar la imagen que subió el usuario
    # imagen = models.ImageField(upload_to='historial_fotos/')

    def __str__(self):
        return f"{self.usuario.username} - {self.nombre_plato} ({self.fecha_analisis.strftime('%d/%m/%Y')})"
    
class HistorialAnalisis(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    plato = models.CharField(max_length=200)
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.plato} - {self.usuario.username}"