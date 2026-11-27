import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Challenge 4: Binarización con umbral binario simple y método de Otsu.

# ── Umbral manual ──────────────────────────────────────────────────────────────
# Cambia este valor (0–255) para aplicarlo a todas las imágenes.
UMBRAL_MANUAL = 100
# ──────────────────────────────────────────────────────────────────────────────


def cargar_ejemplos(carpeta_dataset):
    """Lee la primera imagen de cada clase y la devuelve en escala de grises."""
    ejemplos = {}
    for clase in sorted(os.listdir(carpeta_dataset)):
        carpeta_clase = os.path.join(carpeta_dataset, clase)
        if not os.path.isdir(carpeta_clase):
            continue
        for archivo in sorted(os.listdir(carpeta_clase)):
            if not archivo.lower().endswith(".jpg"):
                continue
            ruta = os.path.join(carpeta_clase, archivo)
            imagen_bgr = cv2.imread(ruta)
            if imagen_bgr is None:
                continue
            imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
            ejemplos[clase] = imagen_gris
            break
    return ejemplos


def binarizar_manual(imagen_gris):
    """Umbral binario simple con el valor fijado en UMBRAL_MANUAL."""
    _, binaria = cv2.threshold(imagen_gris, UMBRAL_MANUAL, 255, cv2.THRESH_BINARY)
    return binaria


def binarizar_otsu(imagen_gris):
    """Umbral de Otsu: OpenCV calcula el valor óptimo automáticamente."""
    umbral, binaria = cv2.threshold(
        imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return int(umbral), binaria


def mostrar_comparacion(ejemplos):
    """Una fila por clase con 4 columnas:
    imagen gris | histograma con ambos umbrales | binaria manual | binaria Otsu."""
    n = len(ejemplos)
    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))

    if n == 1:
        axes = [axes]

    for fila, (clase, imagen_gris) in enumerate(ejemplos.items()):
        u_otsu, bin_otsu = binarizar_otsu(imagen_gris)
        bin_manual = binarizar_manual(imagen_gris)

        print(
            f"{clase:8s} → umbral manual: {UMBRAL_MANUAL:3d} | umbral Otsu: {u_otsu:3d}"
        )

        # Columna 0: imagen en escala de grises
        axes[fila][0].imshow(imagen_gris, cmap="gray")
        axes[fila][0].set_title(f"{clase} — gris")
        axes[fila][0].axis("off")

        # Columna 1: histograma con los dos umbrales marcados
        hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
        axes[fila][1].plot(hist, color="gray", linewidth=0.8)
        axes[fila][1].axvline(UMBRAL_MANUAL, color="blue", linewidth=1.5,
                              label=f"Manual ({UMBRAL_MANUAL})")
        axes[fila][1].axvline(u_otsu, color="red", linewidth=1.5,
                              label=f"Otsu ({u_otsu})")
        axes[fila][1].set_title(f"Histograma — {clase}")
        axes[fila][1].set_xlabel("Intensidad")
        axes[fila][1].set_ylabel("Frecuencia")
        axes[fila][1].legend(fontsize=8)

        # Columna 2: binarización con umbral manual
        axes[fila][2].imshow(bin_manual, cmap="gray")
        axes[fila][2].set_title(f"Umbral manual ({UMBRAL_MANUAL})")
        axes[fila][2].axis("off")

        # Columna 3: binarización con Otsu
        axes[fila][3].imshow(bin_otsu, cmap="gray")
        axes[fila][3].set_title(f"Otsu ({u_otsu})")
        axes[fila][3].axis("off")

    fig.suptitle(
        "Challenge 4 — Binarización: umbral manual vs. Otsu",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def ejercicio4():
    """Ejecuta el Challenge 4: binarización con umbral manual y con Otsu."""
    carpeta_dataset = os.path.join("data", "dataset_limpio")
    ejemplos = cargar_ejemplos(carpeta_dataset)
    print(f"Clases cargadas: {list(ejemplos.keys())}\n")
    mostrar_comparacion(ejemplos)
