import os

import numpy as np
import matplotlib.pyplot as plt

from ejercicios.ejercicio1 import rgb_a_gris_numpy


def miAmpliH(P, a, b, L):
    u = np.array(P, dtype=np.float64)
    X = L * (u - a) / (b - a)
    return np.clip(X, 0, L)


def micuadrada(P, L, O):
    u = np.array(P, dtype=np.float64)
    if O == 0:   # cuadrada (oscurece)
        X = (u ** 2) / L
    else:        # raíz cuadrada (aclara)
        X = np.sqrt(L * u)
    return X


def obtener_imagenes():
    subexpuestas = [
        "cats/0_0325.jpg",
        "cats/0_0501.jpg",
        "cats/0_0461.jpg",
        "cats/0_0395.jpg",
        "dogs/1_0887.jpg",
        "dogs/1_0242.jpg",
        "dogs/1_0352.jpg",
        "dogs/1_0534.jpg",
        "snakes/2_0313.jpg",
        "snakes/2_0476.jpg",
    ]

    sobreexpuestas = [
        "cats/0_0308.jpg",
        "cats/0_0097.jpg",
        "cats/0_0883.jpg",
        "cats/0_0885.jpg",
        "dogs/1_0603.jpg",
        "dogs/1_0538.jpg",
        "dogs/1_0677.jpg",
        "dogs/1_0186.jpg",
        "snakes/2_0393.jpg",
        "snakes/2_0437.jpg",
    ]

    bajo_contraste = [
        "cats/0_0688.jpg",
        "cats/0_0739.jpg",
        "cats/0_0591.jpg",
        "cats/0_0044.jpg",
        "dogs/1_0388.jpg",
        "dogs/1_0748.jpg",
        "dogs/1_0630.jpg",
        "dogs/1_0707.jpg",
        "snakes/2_0797.jpg",
        "snakes/2_0841.jpg",
    ]

    return subexpuestas, sobreexpuestas, bajo_contraste


def procesar_grupo(rutas, algoritmo_a, algoritmo_b, titulo_grupo, nombre_b):
    """Dos figuras por grupo (5 imágenes cada una), 6 columnas por fila."""
    mitades = [rutas[:5], rutas[5:]]
    for parte, mitad in enumerate(mitades):
        fig, axes = plt.subplots(5, 6, figsize=(18, 10))
        fig.suptitle(f"{titulo_grupo} ({parte + 1}/2)", fontsize=14)

        for i, ruta in enumerate(mitad):
            ruta_completa = os.path.join("data", "dataset_limpio", ruta)
            img_rgb = plt.imread(ruta_completa)

            # plt.imread devuelve 0-1 para JPEG, convertir a 0-255
            img_rgb_255 = (img_rgb * 255).astype(np.float32) if img_rgb.max() <= 1.0 else img_rgb.astype(np.float32)
            img_gris = rgb_a_gris_numpy(img_rgb_255)   # float32, rango 0-255

            img_a = np.clip(algoritmo_a(img_gris), 0, 255)
            img_b = np.clip(algoritmo_b(img_gris), 0, 255)
            nombre = os.path.basename(ruta)

            # imshow espera 0-255 con cmap gray cuando son flotantes fuera de 0-1,
            # pasamos vmin/vmax para forzar la escala correcta
            axes[i, 0].imshow(img_gris, cmap="gray", vmin=0, vmax=255)
            axes[i, 0].set_title(f"Original\n{nombre}"); axes[i, 0].axis("off")

            axes[i, 1].hist(img_gris.ravel(), 256, range=(0, 255))
            axes[i, 1].set_title("Hist. Original")

            axes[i, 2].imshow(img_a, cmap="gray", vmin=0, vmax=255)
            axes[i, 2].set_title("miAmpliH"); axes[i, 2].axis("off")

            axes[i, 3].hist(img_a.ravel(), 256, range=(0, 255))
            axes[i, 3].set_title("Hist. miAmpliH")

            axes[i, 4].imshow(img_b, cmap="gray", vmin=0, vmax=255)
            axes[i, 4].set_title(nombre_b); axes[i, 4].axis("off")

            axes[i, 5].hist(img_b.ravel(), 256, range=(0, 255))
            axes[i, 5].set_title(f"Hist. {nombre_b}")

        plt.tight_layout()


def ejercicio2():
    subexpuestas, sobreexpuestas, bajo_contraste = obtener_imagenes()

    # Subexpuestas: estira [0,100] al maximo y raiz cuadrada aclara
    procesar_grupo(
        subexpuestas,
        lambda img: miAmpliH(img, 0, 100, 255),
        lambda img: micuadrada(img, 255, 1),
        "Subexpuesta",
        "micuadrada (raiz)",
    )

    # Sobreexpuestas: comprime techo de 220 a 255 y cuadrada oscurece
    procesar_grupo(
        sobreexpuestas,
        lambda img: miAmpliH(img, 0, 220, 255),
        lambda img: micuadrada(img, 255, 0),
        "Sobreexpuesta",
        "micuadrada (cuadrada)",
    )

    # Bajo contraste: estira automatico al rango real y raiz cuadrada
    procesar_grupo(
        bajo_contraste,
        lambda img: miAmpliH(img, img.min(), img.max(), 255),
        lambda img: micuadrada(img, 255, 1),
        "Bajo contraste",
        "micuadrada (raiz)",
    )

    plt.show()
