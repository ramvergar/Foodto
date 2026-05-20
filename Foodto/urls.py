from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # 1. Pantalla principal (Botones de Login / Register)
    path('', views.inicio_view, name='inicio'),
    
    # 2. El Escáner de la IA (Tu index.html actual)
    path('escaner/', views.escaner_view, name='escaner'),
    
    # 3. Autenticación
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('registro/', views.registro, name='registro'),
    
    # 4. Perfil e Historial
    path('perfil/', views.perfil, name='perfil'),
]