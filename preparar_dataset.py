import os
import random
import cv2
import pandas as pd
from tqdm import tqdm

# 1. RUTAS LOCALES
BASE_PATH = r"C:\Users\Sara\Desktop\TFG\archive\real_vs_fake\real-vs-fake"
ORIGINAL_REAL_DIR = os.path.join(BASE_PATH, "train", "real")
ORIGINAL_FAKE_DIR = os.path.join(BASE_PATH, "train", "fake")

OUTPUT_BASE_DIR = r"C:\Users\Sara\Desktop\TFG_datasets\datasets_procesados"

# 2. PARÁMETROS
SIZES = [(64, 64), (128, 128), (256, 256)]
SAMPLE_COUNTS = [1000, 5000, 10000]

random.seed(42)

def get_sampled_files(directory, count):
    all_files = os.listdir(directory)
    return random.sample(all_files, min(count, len(all_files)))

max_samples = max(SAMPLE_COUNTS)
selected_real_files = get_sampled_files(ORIGINAL_REAL_DIR, max_samples)
selected_fake_files = get_sampled_files(ORIGINAL_FAKE_DIR, max_samples)


# 3. BUCLE DE GENERACIÓN DE IMÁGENES + CSV
for img_size in SIZES: # Bucle externo: recorre cada uno de los tamaños de imagen definidos en SIZES
    for count in SAMPLE_COUNTS: # Bucle interno: recorre cada cantidad de muestras definida (1000, 5000, 10000)
        # Construir rutas 
        target_dir = os.path.join(OUTPUT_BASE_DIR, f"size_{img_size[0]}x{img_size[1]}", f"sample_{count//1000}k") # Crea una ruta de salida específica para esta combinación 
        target_real_dir = os.path.join(target_dir, "real") # Define la subcarpeta de destino para las imágenes reales procesadas
        target_fake_dir = os.path.join(target_dir, "fake") # Define la subcarpeta de destino para las imágenes falsas procesadas
        
        os.makedirs(target_real_dir, exist_ok=True) # Crea físicamente la carpeta de destino para reales 
        os.makedirs(target_fake_dir, exist_ok=True) # Crea físicamente la carpeta de destino para falsas 
        
        records = [] # Inicializa una lista vacía para almacenar los metadatos (ruta, nombre, etiqueta) de cada imagen procesada

        # Procesar REALES (Etiqueta 0)
        for fname in tqdm(selected_real_files[:count], desc="Procesando Reales"): # Recorre el subconjunto de archivos reales según el 'count' actual
            dst = os.path.join(target_real_dir, fname) # Define la ruta de destino exacta para este archivo específico
            img = cv2.imread(os.path.join(ORIGINAL_REAL_DIR, fname)) # Lee la imagen original desde el disco usando OpenCV
            
            if img is not None: # Comprueba que la imagen se haya leído correctamente y no esté corrupta
                cv2.imwrite(dst, cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)) # Redimensiona la imagen al tamaño actual ('img_size') usando interpolación por área y la guarda en el destino
                records.append({'filepath': dst, 'filename': fname, 'label': 0, 'label_str': 'real'}) # Añade un diccionario con los datos del archivo y su etiqueta numérica (0 para real) a la lista

        # Procesar FAKES (Etiqueta 1)
        for fname in tqdm(selected_fake_files[:count], desc="Procesando Fakes"):
            dst = os.path.join(target_fake_dir, fname) 
            img = cv2.imread(os.path.join(ORIGINAL_FAKE_DIR, fname)) 
            
            if img is not None: # Verifica que la imagen sea válida
                cv2.imwrite(dst, cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)) 
                records.append({'filepath': dst, 'filename': fname, 'label': 1, 'label_str': 'fake'})

        # Crear y guardar CSV
        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True) # Convierte la lista de registros en un DataFrame de Pandas, lo mezcla de forma completamente aleatoria (shuffle) y reinicia los índices
        df.to_csv(os.path.join(target_dir, "dataset_info.csv"), index=False) # Guarda el DataFrame resultante como un archivo CSV llamado 'dataset_info.csv' dentro de la carpeta de esa combinación, sin incluir el índice

print("\n¡Proceso completado!")
