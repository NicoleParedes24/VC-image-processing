import os

import cv2
import matplotlib.pyplot as plt



def ecualizar_global(imagen_gris):
    """Aplica ecualización  con OpenCV."""
    return cv2.equalizeHist(imagen_gris)


def ecualizar_clahe(imagen_gris):
    """Aplica ecualización CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(imagen_gris)


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

            imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
            imagenes.append((clase, imagen_gris))
            cantidad += 1
            if clase not in ejemplos:
                ejemplos[clase] = imagen_gris

        print(f"{clase}: {cantidad} imágenes")

    return imagenes, ejemplos


def calcular_contraste_por_clase(imagenes):
    """Calcula el contraste promedio antes y después de ecualizar."""
    acumulado = {}

    for clase, imagen_gris in imagenes:
        if clase not in acumulado:
            acumulado[clase] = [0.0, 0.0, 0.0, 0]

        eq = ecualizar_global(imagen_gris)
        clahe = ecualizar_clahe(imagen_gris)

        acumulado[clase][0] += float(imagen_gris.std())
        acumulado[clase][1] += float(eq.std())
        acumulado[clase][2] += float(clahe.std())
        acumulado[clase][3] += 1

    print("Contraste promedio (desviación estándar) por clase:")
    for clase, (orig, eq, clahe, n) in acumulado.items():
        print(
            f"  {clase:8s} -> Original: {orig / n:5.2f} | "
            f"equalizeHist: {eq / n:5.2f} | CLAHE: {clahe / n:5.2f}"
        )

#para hacer comparacion

def mostrar_comparacion(ejemplos):
    
    plt.figure(figsize=(13, 3.5 * len(ejemplos)))
    posicion = 1

    for clase, imagen_gris in ejemplos.items():
        eq = ecualizar_global(imagen_gris)
        clahe = ecualizar_clahe(imagen_gris)

        plt.subplot(len(ejemplos), 4, posicion)
        plt.imshow(imagen_gris, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{clase} original")
        plt.axis("off")

        plt.subplot(len(ejemplos), 4, posicion + 1)
        plt.imshow(eq, cmap="gray", vmin=0, vmax=255)
        plt.title("equalizeHist")
        plt.axis("off")

        plt.subplot(len(ejemplos), 4, posicion + 2)
        plt.imshow(clahe, cmap="gray", vmin=0, vmax=255)
        plt.title("CLAHE")
        plt.axis("off")

        hist_orig = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
        hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256]).flatten()
        hist_clahe = cv2.calcHist([clahe], [0], None, [256], [0, 256]).flatten()

        plt.subplot(len(ejemplos), 4, posicion + 3)
        plt.plot(hist_orig, color="gray", linewidth=1, label="Original")
        plt.plot(hist_eq, color="blue", linewidth=0.9, label="equalizeHist")
        plt.plot(hist_clahe, color="orange", linewidth=1, label="CLAHE")
        plt.title(f"Histograma - {clase}")
        plt.xlim(0, 255)
        plt.legend(fontsize=8)

        posicion += 4

    plt.tight_layout()
    plt.show()


def ejercicio3():
    carpeta_dataset = os.path.join("data", "dataset_limpio")
    imagenes, ejemplos = cargar_imagenes(carpeta_dataset)
    print(f"Total: {len(imagenes)} imágenes\n")

    calcular_contraste_por_clase(imagenes)

    print(
        "\nDiscusión:\n"
        "- equalizeHist distribuye el histograma en 0:255 y mejora los "
        "detalles en zonas oscuras, pero puede exagerar el contraste o ruido en fondos "
        "lisos y oscurecer tonos medios en zonas sobreexpuestas.\n"
        "- CLAHE mejora el contraste de forma local y más natural: resalta el pelaje en "
        "cats y dogs sin arruinar el fondo, y destaca mejor las escamas en snakes."
    )
    mostrar_comparacion(ejemplos)

if __name__ == "__main__":
    ejercicio3()
