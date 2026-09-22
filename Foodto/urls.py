from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.landing_view, name='landing'),

    path('escaner/', views.escaner_view, name='escaner'),

    path('login/', auth_views.LoginView.as_view(
        template_name='Foodto/login.html'
    ), name='login'),

    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    path('registro/', views.registro, name='registro'),

    path('perfil/', views.perfil, name='perfil'),
]