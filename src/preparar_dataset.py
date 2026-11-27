"""
Prepara el dataset limpio del proyecto a partir del dataset crudo de Kaggle.

Dataset original: "Animals (Cats, Dogs, and Snakes)", Omar Rehan, licencia CC0.
https://www.kaggle.com/datasets/aiomarrehan/animals-cats-dogs-and-snakes

Requisitos del proyecto (Parcial 1):
  - hasta 4 clases, imágenes RGB, al menos 200 imágenes (50 a 70 por clase);
  - el Challenge 2 necesita al menos 10 imágenes subexpuestas, 10 sobreexpuestas
    y 10 con poco contraste.

Pasos:
  1. Leer las 3000 imágenes crudas y calcular sus propiedades.
  2. Limpiar: quitar imágenes ilegibles, que no son RGB de verdad (los tres
     canales iguales o casi iguales), duplicadas (hash MD5 y dHash) y muy borrosas.
  3. Etiquetar la exposición de cada imagen a partir de su histograma en gris.
  4. Elegir 70 imágenes por clase con semilla fija. En cada clase se garantizan
     4 subexpuestas, 4 sobreexpuestas y 4 de bajo contraste (12 de cada tipo en
     total); el resto se toma al azar entre las imágenes limpias.
  5. Copiar los archivos sin modificarlos (mismos bytes JPEG) y guardar los CSV.

Uso (desde la carpeta royecto/):
    python src/preparar_dataset.py
"""

import csv
import hashlib
import os
import random
import shutil

import cv2
import numpy as np

# ---------------------------------------------------------------- parámetros
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRUDO = os.path.join(RAIZ, "data", "raw", "Animals")
SALIDA = os.path.join(RAIZ, "data", "dataset_limpio")

SEMILLA = 42
POR_CLASE = 70
MINIMO_POR_TIPO = 4          # por clase y por tipo de exposición -> 12 en total

UMBRAL_SUBEXPUESTA = 70      # brillo medio en gris por debajo de este valor
UMBRAL_SOBREEXPUESTA = 185   # brillo medio en gris por encima de este valor
UMBRAL_BAJO_CONTRASTE = 35   # desviación estándar en gris por debajo de este valor
UMBRAL_CASI_GRIS = 2.0       # diferencia media |B - R| por debajo: imagen sin color
UMBRAL_BORROSA = 50.0        # varianza del laplaciano por debajo: imagen borrosa
DISTANCIA_DHASH = 0          # bits distintos para considerar dos imágenes duplicadas

# Imágenes descartadas tras revisar a ojo las hojas de contacto de la muestra:
# no son fotografías del animal o tienen grandes márgenes blancos que alteran
# el histograma y la binarización.
EXCLUSION_MANUAL = {
    "snakes/2_0160.jpg": "página de documento con texto",
    "snakes/2_0731.jpg": "infografía con texto y fotos de piel",
    "snakes/2_0236.jpg": "estampado sobre un cojín, no es foto del animal",
    "snakes/2_0303.jpg": "diagrama con escala de medidas",
    "snakes/2_0471.jpg": "sello postal ilustrado",
    "snakes/2_0724.jpg": "portada con texto",
    "snakes/2_0873.jpg": "mapa de distribución",
    "snakes/2_0621.jpg": "mapa de distribución",
    "snakes/2_0918.jpg": "foto con texto superpuesto",
    "cats/0_0728.jpg": "foto pequeña con gran margen blanco",
    "dogs/1_0366.jpg": "franja blanca grande a un lado",
    "dogs/1_0651.jpg": "foto pequeña con gran margen blanco",
}


def leer_imagen(ruta):
    """Lee la imagen con OpenCV (BGR) aunque la ruta tenga tildes."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_UNCHANGED), datos.tobytes()


def dhash(gris):
    """Hash de diferencias de 64 bits: detecta copias aunque cambie la compresión."""
    peq = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits(peq[:, 1:] > peq[:, :-1]).tobytes()


def tipo_exposicion(media, desviacion):
    """Clasifica la imagen según su histograma (media y dispersión de grises)."""
    if media < UMBRAL_SUBEXPUESTA:
        return "subexpuesta"
    if media > UMBRAL_SOBREEXPUESTA:
        return "sobreexpuesta"
    if desviacion < UMBRAL_BAJO_CONTRASTE:
        return "bajo_contraste"
    return "normal"


def analizar_crudo():
    """Recorre el dataset crudo y devuelve una fila de propiedades por imagen."""
    filas = []
    for clase in sorted(os.listdir(CRUDO)):
        carpeta = os.path.join(CRUDO, clase)
        for nombre in sorted(os.listdir(carpeta)):
            ruta = os.path.join(carpeta, nombre)
            img, datos = leer_imagen(ruta)
            fila = {"clase": clase, "archivo": nombre, "ruta": ruta,
                    "peso_kb": round(len(datos) / 1024, 2),
                    "md5": hashlib.md5(datos).hexdigest()}
            if img is None:
                fila["motivo"] = "ilegible"
                filas.append(fila)
                continue
            canales = 1 if img.ndim == 2 else img.shape[2]
            gris = img if canales == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dif_color = 0.0 if canales == 1 else float(
                np.abs(img[..., 0].astype(np.int16) - img[..., 2]).mean())
            fila.update({
                "alto": img.shape[0], "ancho": img.shape[1], "canales": canales,
                "bits_por_canal": img.dtype.itemsize * 8,
                "media_gris": round(float(gris.mean()), 2),
                "desv_gris": round(float(gris.std()), 2),
                "nitidez_laplaciano": round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 1),
                "dif_color": round(dif_color, 2),
                "dhash": dhash(gris),
            })
            fila["exposicion"] = tipo_exposicion(fila["media_gris"], fila["desv_gris"])
            filas.append(fila)
    return filas


def limpiar(filas):
    """Marca con 'motivo' las imágenes que se descartan; las demás quedan limpias."""
    vistos_md5, vistos_dhash = {}, []
    for f in filas:
        if f.get("motivo"):
            continue
        if f["clase"] + "/" + f["archivo"] in EXCLUSION_MANUAL:
            f["motivo"] = "revisión manual: " + EXCLUSION_MANUAL[f["clase"] + "/" + f["archivo"]]
        elif f["canales"] != 3 or f["dif_color"] < UMBRAL_CASI_GRIS:
            f["motivo"] = "sin color (gris guardado como RGB)"
        elif f["nitidez_laplaciano"] < UMBRAL_BORROSA:
            f["motivo"] = "borrosa"
        elif f["md5"] in vistos_md5:
            f["motivo"] = "duplicado exacto de " + vistos_md5[f["md5"]]
        else:
            h = np.unpackbits(np.frombuffer(f["dhash"], np.uint8))
            copia = next((n for n, h2 in vistos_dhash
                          if np.count_nonzero(h != h2) <= DISTANCIA_DHASH), None)
            if copia:
                f["motivo"] = "casi duplicado de " + copia
        if not f.get("motivo"):
            nombre = f["clase"] + "/" + f["archivo"]
            vistos_md5[f["md5"]] = nombre
            vistos_dhash.append((nombre, np.unpackbits(np.frombuffer(f["dhash"], np.uint8))))
    return [f for f in filas if not f.get("motivo")], [f for f in filas if f.get("motivo")]


def muestrear(limpias):
    """Elige POR_CLASE imágenes por clase garantizando los tres tipos de exposición.

    Cada imagen recibe un número aleatorio fijo (derivado de la semilla y su nombre)
    y se toman las de número más bajo. Así, si se descarta una imagen a mano, solo
    entra su reemplazo y el resto de la muestra no cambia.
    """
    def orden(f):
        return random.Random(f"{SEMILLA}/{f['clase']}/{f['archivo']}").random()

    elegidas = []
    for clase in sorted({f["clase"] for f in limpias}):
        de_clase = sorted((f for f in limpias if f["clase"] == clase), key=orden)
        tomadas = []
        for tipo in ("subexpuesta", "sobreexpuesta", "bajo_contraste"):
            candidatas = [f for f in de_clase if f["exposicion"] == tipo]
            if len(candidatas) < MINIMO_POR_TIPO:
                raise ValueError(f"{clase}: solo hay {len(candidatas)} imágenes {tipo}")
            tomadas += candidatas[:MINIMO_POR_TIPO]
        resto = [f for f in de_clase if f not in tomadas]
        tomadas += resto[:POR_CLASE - len(tomadas)]
        elegidas += sorted(tomadas, key=lambda f: f["archivo"])
    return elegidas


def guardar_csv(ruta, filas, columnas):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def main():
    filas = analizar_crudo()
    limpias, descartadas = limpiar(filas)
    muestra = muestrear(limpias)

    # Se rehace la carpeta de salida para que el resultado sea siempre el mismo.
    if os.path.isdir(SALIDA):
        shutil.rmtree(SALIDA)
    for f in muestra:
        destino = os.path.join(SALIDA, f["clase"])
        os.makedirs(destino, exist_ok=True)
        shutil.copy2(f["ruta"], os.path.join(destino, f["archivo"]))

    columnas = ["clase", "archivo", "ancho", "alto", "canales", "bits_por_canal",
                "peso_kb", "media_gris", "desv_gris", "nitidez_laplaciano", "exposicion"]
    guardar_csv(os.path.join(RAIZ, "data", "metadata.csv"), muestra, columnas)
    guardar_csv(os.path.join(RAIZ, "data", "descartadas.csv"), descartadas,
                ["clase", "archivo", "motivo"])

    print(f"Crudo: {len(filas)} | limpias: {len(limpias)} | descartadas: {len(descartadas)}")
    for clase in sorted({f["clase"] for f in muestra}):
        m = [f for f in muestra if f["clase"] == clase]
        cuenta = {t: sum(f["exposicion"] == t for f in m)
                  for t in ("subexpuesta", "sobreexpuesta", "bajo_contraste", "normal")}
        print(f"  {clase}: {len(m)} imágenes {cuenta}")
    print(f"Muestra final: {len(muestra)} imágenes en {SALIDA}")


if __name__ == "__main__":
    main()
