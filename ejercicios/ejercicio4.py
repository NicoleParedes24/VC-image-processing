import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Challenge 4: Binarización con umbral binario simple y método de Otsu.

UMBRAL_MANUAL = 110


def cargar_imagenes(carpeta_dataset):
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


def binarizar_manual(imagen_gris):
    "Umbral binario simple con el valor fijado en UMBRAL_MANUAL."
    binaria = cv2.threshold(imagen_gris, UMBRAL_MANUAL, 255, cv2.THRESH_BINARY)
    return binaria


def binarizar_otsu(imagen_gris):
    "Umbral de Otsu: OpenCV calcula el valor óptimo automáticamente."
    umbral, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(umbral), binaria


def mostrar_comparacion(ejemplos):
    plt.figure(figsize=(14, 4 * len(ejemplos)))
    posicion = 1

    for clase, imagen_rgb in ejemplos.items():
        imagen_gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
        bin_manual = binarizar_manual(imagen_gris)
        u_otsu, bin_otsu = binarizar_otsu(imagen_gris)

        print(f"{clase:8s} → umbral manual: {UMBRAL_MANUAL:3d} | umbral Otsu: {u_otsu:3d}")

        # imagen original
        plt.subplot(len(ejemplos), 4, posicion)
        plt.imshow(imagen_rgb)
        plt.title(f"{clase} original")
        plt.axis("off")

        # histograma con ambos umbrales
        hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
        plt.subplot(len(ejemplos), 4, posicion + 1)
        plt.plot(hist, color="gray", linewidth=0.8)
        plt.axvline(UMBRAL_MANUAL, color="blue", linewidth=1.5,
                    label=f"Manual ({UMBRAL_MANUAL})")
        plt.axvline(u_otsu, color="red", linewidth=1.5,
                    label=f"Otsu ({u_otsu})")
        plt.title(f"Histograma — {clase}")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.legend(fontsize=8)

        # binarización con umbral manual
        plt.subplot(len(ejemplos), 4, posicion + 2)
        plt.imshow(bin_manual, cmap="gray")
        plt.title(f"Umbral manual ({UMBRAL_MANUAL})")
        plt.axis("off")

        # Columna 4: binarización con Otsu
        plt.subplot(len(ejemplos), 4, posicion + 3)
        plt.imshow(bin_otsu, cmap="gray")
        plt.title(f"Otsu ({u_otsu})")
        plt.axis("off")

        posicion += 4

    plt.suptitle(
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def ejercicio4():
    carpeta_dataset = os.path.join("data", "dataset_limpio")
    imagenes, ejemplos = cargar_imagenes(carpeta_dataset)
    print(f"Total: {len(imagenes)} imágenes\n")
    mostrar_comparacion(ejemplos)
