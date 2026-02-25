from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from Foodto import views  # Importamos nuestras vistas

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.escaner_view, name='escaner'), # La página principal
]

# Esto permite que Django sirva las fotos que suben los usuarios durante el desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)