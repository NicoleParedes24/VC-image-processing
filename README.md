# VC-image-processing

Proyecto del Parcial 1 de Visión por Computador: **binarización y modificación de contraste** con Python y OpenCV, sobre un dataset real de Kaggle.

## Dataset

**Animals (Cats, Dogs, and Snakes)**, de Omar Rehan, con licencia CC0.
https://www.kaggle.com/datasets/aiomarrehan/animals-cats-dogs-and-snakes

| | Dataset original | Muestra usada en el proyecto |
|---|---|---|
| Clases | 3 (cats, dogs, snakes) | 3 |
| Imágenes | 3000 (1000 por clase) | 210 (70 por clase) |
| Tamaño | 256 × 256 px | 256 × 256 px |
| Formato | JPEG | JPEG (archivos sin modificar) |
| Canales / profundidad | RGB, 8 bits por canal (24 bits) | RGB, 8 bits por canal (24 bits) |

La ficha de Kaggle dice que las imágenes miden 224 × 224, pero al leerlas todas miden **256 × 256**. La ficha no indica de dónde se obtuvieron originalmente las imágenes; solo dice que se limpiaron y redimensionaron.

### Cómo se armó la muestra limpia

El script [`src/preparar_dataset.py`](src/preparar_dataset.py) parte de las 3000 imágenes y:

1. **Descarta 60 imágenes** (el motivo de cada una está en [`data/descartadas.csv`](data/descartadas.csv)):
   - 20 que no tienen color: son imágenes en gris guardadas como RGB (diferencia media entre canales < 2).
   - 24 borrosas: varianza del laplaciano < 50.
   - 4 duplicadas: una por MD5 idéntico y tres por dHash idéntico.
   - 12 que, revisadas a ojo, no son fotos del animal: mapas, sellos, portadas, diagramas o imágenes con grandes márgenes blancos.
2. **Etiqueta la exposición** de cada imagen con su histograma en escala de grises:
   - subexpuesta: brillo medio < 70;
   - sobreexpuesta: brillo medio > 185;
   - bajo contraste: desviación estándar < 35;
   - normal: el resto.
3. **Elige 70 imágenes por clase** con una semilla fija (42). En cada clase se garantizan al menos 4 de cada tipo de exposición, para que el Challenge 2 tenga al menos 10 subexpuestas, 10 sobreexpuestas y 10 de bajo contraste.

Resultado: **210 imágenes**, de las cuales 17 son subexpuestas, 12 sobreexpuestas, 14 de bajo contraste y 167 normales. Las propiedades de cada imagen están en [`data/metadata.csv`](data/metadata.csv).

## Estructura

```
data/
  dataset_limpio/{cats,dogs,snakes}/   muestra de 210 imágenes (se sube al repo, 3 MB)
  metadata.csv                         propiedades y tipo de exposición de cada imagen
  descartadas.csv                      imágenes descartadas y el motivo
  raw/                                 dataset original (no se sube; ver abajo)
src/
  preparar_dataset.py                  limpieza y muestreo reproducible
```

## Cómo reproducir la muestra

```bash
pip install -r requirements.txt
# Descargar el zip desde Kaggle y descomprimirlo en data/raw/ (debe quedar data/raw/Animals/{cats,dogs,snakes})
python src/preparar_dataset.py
```

## Retos del proyecto

1. Conversión a escala de grises: implementación propia con NumPy comparada con `cv2.cvtColor`.
2. Ajuste de contraste con al menos dos algoritmos.
3. Ecualización de histograma (`cv2.equalizeHist` y CLAHE).
4. Binarización con umbral manual y con Otsu.
