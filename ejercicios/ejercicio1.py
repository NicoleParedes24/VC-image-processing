import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Ejercicio 1: convertir imágenes RGB a gris y comparar NumPy con OpenCV.
def rgb_a_gris_numpy(imagen_rgb):
    """Convierte una imagen RGB a gris usando la fórmula ponderada."""
    rojo = imagen_rgb[:, :, 0].astype(np.float32)
    verde = imagen_rgb[:, :, 1].astype(np.float32)
    azul = imagen_rgb[:, :, 2].astype(np.float32)
    return 0.299 * rojo + 0.587 * verde + 0.114 * azul


def cargar_imagenes(carpeta_dataset):
    """Lee los JPG del dataset y guarda un ejemplo de cada clase."""
    imagenes = []
    ejemplos = {}

    for clase in sorted(os.listdir(carpeta_dataset)):
        carpeta_clase = os.path.join(carpeta_dataset, clase)
        if not os.path.isdir(carpeta_clase):
            continue

        cantidad = 0
        for archivo in sorted(os.listdir(carpeta_clase)):
            if not archivo.lower().endswith(".jpg"):
                continue

            ruta = os.path.join(carpeta_clase, archivo)
            imagen_bgr = cv2.imread(ruta)
            if imagen_bgr is None:
                continue

            imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
            imagenes.append((clase, imagen_rgb))
            cantidad += 1
            if clase not in ejemplos:
                ejemplos[clase] = imagen_rgb

        print(f"{clase}: {cantidad} imágenes")

    return imagenes, ejemplos


def calcular_error_medio(imagenes):
    """Compara las conversiones NumPy y OpenCV en todos los píxeles."""
    error_total = 0
    pixeles_totales = 0

    for clase, imagen_rgb in imagenes:
        gris_numpy = rgb_a_gris_numpy(imagen_rgb)
        gris_opencv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
        error_total += np.abs(gris_numpy - gris_opencv).sum()
        pixeles_totales += gris_numpy.size

    return error_total / pixeles_totales


def medir_tiempo_numpy(imagenes):
    """Mide cuánto tarda NumPy en convertir todas las imágenes."""
    inicio = time.time()
    for clase, imagen_rgb in imagenes:
        rgb_a_gris_numpy(imagen_rgb)
    return time.time() - inicio


def medir_tiempo_opencv(imagenes):
    """Mide cuánto tarda OpenCV en convertir todas las imágenes."""
    inicio = time.time()
    for clase, imagen_rgb in imagenes:
        cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    return time.time() - inicio


def mostrar_comparacion(ejemplos):
    """Muestra el original y las dos versiones en escala de grises."""
    plt.figure(figsize=(9, 9))
    posicion = 1

    for clase, imagen_rgb in ejemplos.items():
        gris_numpy = rgb_a_gris_numpy(imagen_rgb)
        gris_opencv = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)

        plt.subplot(len(ejemplos), 3, posicion)
        plt.imshow(imagen_rgb)
        plt.title(f"{clase} original")
        plt.axis("off")

        plt.subplot(len(ejemplos), 3, posicion + 1)
        plt.imshow(gris_numpy, cmap="gray")
        plt.title("Gris NumPy")
        plt.axis("off")

        plt.subplot(len(ejemplos), 3, posicion + 2)
        plt.imshow(gris_opencv, cmap="gray")
        plt.title("Gris OpenCV")
        plt.axis("off")

        posicion += 3

    plt.tight_layout()
    plt.show()


def ejercicio1():
    """Ejecuta todas las partes del Ejercicio 1."""
    carpeta_dataset = os.path.join("data", "dataset_limpio")
    imagenes, ejemplos = cargar_imagenes(carpeta_dataset)
    print(f"Total: {len(imagenes)} imágenes\n")

    error = calcular_error_medio(imagenes)
    tiempo_numpy = medir_tiempo_numpy(imagenes)
    tiempo_opencv = medir_tiempo_opencv(imagenes)

    print(f"Error absoluto medio: {error:.4f}")
    print(f"Tiempo NumPy: {tiempo_numpy:.4f} segundos")
    print(f"Tiempo OpenCV: {tiempo_opencv:.4f} segundos")

    mostrar_comparacion(ejemplos)
