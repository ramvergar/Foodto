# 🍔 Foodto - Sistema de Análisis Visual de Alérgenos

Este proyecto es una aplicación web Full-Stack desarrollada con **Django** e **Inteligencia Artificial (PyTorch)**. Permite a los usuarios subir imágenes de platos de comida para, mediante una red neuronal convolucional (ResNet18 con Fine-Tuning), identificar el plato, cruzar los datos y devolver una evaluación del nivel de riesgo y los alérgenos presentes.

---

## ⚙️ Requisitos Previos

Para ejecutar este proyecto en un nuevo equipo, asegúrate de tener instalado:
* **Python 3.8 o superior** (Durante la instalación en Windows, es vital marcar la casilla "Add Python to PATH").
* **Git** (Opcional, si vas a clonar el repositorio).

---

## 🚀 Guía de Instalación y Despliegue Local (Windows)

Sigue estos pasos en la terminal (Símbolo del sistema o PowerShell) para poner en marcha el proyecto desde cero:

### 1. Obtener el proyecto
Descarga la carpeta del proyecto en tu equipo. Abre una terminal y navega hasta la raíz del proyecto (la carpeta donde se encuentra el archivo manage.py).

### 2. Crear y Activar el Entorno Virtual
Para no interferir con el entorno global de Python del sistema, creamos un entorno aislado. Ejecuta estos dos comandos:

    python -m venv venv
    venv\Scripts\activate

*(Sabrás que ha funcionado porque verás un `(venv)` al principio de la línea de comandos).*

### 3. Instalar las Dependencias
Con el entorno virtual activado, instala todas las librerías necesarias (Django, PyTorch, etc.):

    pip install -r requirements.txt

### 4. Configurar las Variables de Entorno (.env)
Este proyecto protege sus credenciales por seguridad. Debes crear tu propio archivo de configuración local:
1. En la carpeta raíz, busca el archivo llamado `.env.example`.
2. Haz una copia de ese archivo y cámbiale el nombre para que sea exactamente `.env`
3. Abre el nuevo archivo `.env` con el Bloc de notas o VS Code y rellena las variables con tus credenciales reales:

    GOOGLE_API_KEY=tu_api_key_aqui
    DJANGO_SECRET_KEY=tu_secret_key_aqui
    DEBUG=True

### 5. Inicializar la Base de Datos
Prepara la base de datos interna de Django (SQLite) aplicando las migraciones del sistema:

    python manage.py migrate

### 6. Arrancar el Servidor Local
Una vez está todo instalado y configurado, enciende el motor web y carga la IA en memoria:

    python manage.py runserver

Si todo ha ido bien, la terminal indicará que el servidor está corriendo sin errores. 
Abre tu navegador web y entra en: 👉 http://127.0.0.1:8000/

---

## 📂 Estructura Principal del Proyecto

* `core_tfg/`: Configuración principal de Django.
* `Foodto/`: Aplicación principal.
  * `apps.py`: Contiene el Singleton que carga el modelo .pth en memoria al arrancar.
  * `views.py`: Controlador principal (procesa la imagen y ejecuta la IA).
  * `templates/Foodto/index.html`: Interfaz de usuario minimalista.
* `media/`: Almacenamiento temporal de las imágenes escaneadas.
* `.env.example`: Plantilla de variables de entorno seguras.
* `requirements.txt`: Listado estricto de librerías y dependencias.
