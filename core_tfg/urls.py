from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from Foodto import views  # Importamos nuestras vistas
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.inicio_view, name='inicio'),
    
    # 2. El Escáner de la IA (Tu index.html actual)
    path('escaner/', views.escaner_view, name='escaner'),
    
    # 3. Autenticación
    path('login/', auth_views.LoginView.as_view(template_name='Foodto/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('registro/', views.registro, name='registro'),
    
    # 4. Perfil e Historial
    path('perfil/', views.perfil, name='perfil'),
]

# Esto permite que Django sirva las fotos que suben los usuarios durante el desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)