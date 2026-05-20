import os

CARPETA = "dataset_comida_grande"
total_fotos = 0
clases_con_fotos = 0

if os.path.exists(CARPETA):
    print(f"\n📊 --- RECUENTO DEL DATASET ---")
    for clase in sorted(os.listdir(CARPETA)):
        ruta_clase = os.path.join(CARPETA, clase)
        if os.path.isdir(ruta_clase):
            fotos = len([f for f in os.listdir(ruta_clase) if f.endswith('.jpg')])
            print(f" > {clase.upper()}: {fotos} fotos")
            total_fotos += fotos
            if fotos > 0:
                clases_con_fotos += 1
                
    print("-" * 30)
    print(f"✅ TOTAL ACUMULADO: {total_fotos} imágenes repartidas en {clases_con_fotos} clases.\n")
else:
    print(f"❌ La carpeta '{CARPETA}' no se ha encontrado.")