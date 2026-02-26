# 🍔 Foodto - Sistema de Análisis Visual de Alérgenos

Este proyecto es una aplicación web Full-Stack desarrollada con **Django** e **Inteligencia Artificial (PyTorch)**. Permite a los usuarios subir imágenes de platos de comida para, mediante una red neuronal convolucional (ResNet18 con Fine-Tuning), identificar el plato, cruzar los datos y devolver una evaluación del nivel de riesgo y los alérgenos presentes.

---

## ⚙️ Requisitos Previos

Para ejecutar este proyecto en un nuevo equipo, asegúrate de tener instalado:
* **Python 3.8 o superior** (Asegúrate de marcar la casilla "Add Python to PATH" durante la instalación en Windows).
* **Git** (Opcional, si vas a clonar el repositorio).

---

## 🚀 Guía de Instalación y Despliegue Local

Sigue estos pasos en la terminal de tu sistema para poner en marcha el proyecto desde cero:

### 1. Obtener el proyecto
Clona el repositorio o copia la carpeta del proyecto en tu equipo. Abre una terminal y navega hasta la raíz del proyecto (donde se encuentra el archivo `manage.py`).

### 2. Crear y "Encender" el Entorno Virtual (venv)
Para no interferir con el Python global de tu ordenador, creamos un entorno aislado.

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
