import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path



#Parte 1: Interpolacion bilineal para redimensionar

def redimensionar_imagen(ruta_imagen):
    # 1. Abrir la imagen y convertirla a escala de grises
    imagen = Image.open(ruta_imagen).convert("L")

    # 2. Convertir los píxeles a un tensor de PyTorch
    array = np.array(imagen, dtype=np.float32)
    tensor = torch.from_numpy(array)

    # 3. Añadir las dimensiones 
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # 4. Convertir los píxeles al intervalo [0, 1]
    tensor = tensor / 255.0

    # 5. Redimensionar mediante interpolación bilineal
    tensor = F.interpolate(
        tensor,
        size=(155, 220),
        mode="bilinear",
        align_corners=False
    )

    # 6. Eliminar las dimensiones adicionales
    tensor = tensor.squeeze(0).squeeze(0)

    return tensor
    
#Parte 2: Imagen inversa 
    
def invertir_imagen(imagen):
    imagen_invertida = 1.0 - imagen
    return imagen_invertida
    
    
    #Parte 3: Normalizar las imagenes
#Como la imagen se normaliza a partir de la desviacion estándar, esta debe de medirse de primero
#Dentro de esta fucnción ya se deberian de aplicar las otras dos, para que la desviacion sea respecto 
#a las imagenes ya redimensionadas e inversas

def calcular_desviacion_estandar(rutas_imagenes):
    suma_pixeles = 0.0
    suma_cuadrados = 0.0
    numero_pixeles = 0

    for ruta in rutas_imagenes:

        # Redimensionar la imagen
        imagen = redimensionar_imagen(ruta)

        # Invertir las intensidades
        imagen = invertir_imagen(imagen)

        # Acumular las estadísticas
        suma_pixeles += imagen.sum().item()
        suma_cuadrados += (imagen ** 2).sum().item()

        numero_pixeles += imagen.numel()

    # Calcular la media global
    media = suma_pixeles / numero_pixeles

    # Calcular la desviación estándar poblacional
    varianza = suma_cuadrados / numero_pixeles - media ** 2

    desviacion = max(varianza, 0.0) ** 0.5

    return desviacion
    
#Cuando ya se tiene la desviación la normalizacion es simplemente: 
def normalizar_imagen(imagen, desviacion_estandar):

    imagen_normalizada = imagen / desviacion_estandar

    return imagen_normalizada


#Paso 4: Aplicar la operacion: 
def preprocesar_imagen(ruta_imagen, desviacion_estandar):

    # Paso 1: redimensionar
    imagen = redimensionar_imagen(ruta_imagen)

    # Paso 2: invertir
    imagen = invertir_imagen(imagen)

    # Paso 3: normalizar
    imagen = normalizar_imagen(imagen, desviacion_estandar)
    return imagen
    
#Paso 5: Main

def main():
    # Directorio donde está guardado este script
    CARPETA_PROYECTO = Path(__file__).resolve().parent

    # Carpeta que contiene los datos de CEDAR
    CARPETA_CEDAR = CARPETA_PROYECTO / "data"
    
    # Buscar las imágenes del conjunto CEDAR
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    rutas_imagenes = sorted(
        ruta
        for ruta in CARPETA_CEDAR.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() in extensiones
    )

    if not rutas_imagenes:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {CARPETA_CEDAR.resolve()}"
        )

    print("Número de imágenes:", len(rutas_imagenes))

    # Calcular una sola desviación estándar global
    desviacion = calcular_desviacion_estandar(rutas_imagenes)

    print("Desviación estándar:", desviacion)

    # Procesar una imagen de ejemplo
    imagen = preprocesar_imagen(
        rutas_imagenes[0],
        desviacion
    )

    print("Imagen:", rutas_imagenes[0].name)
    print("Forma:", imagen.shape)
    print("Valor mínimo:", imagen.min().item())
    print("Valor máximo:", imagen.max().item())

    # Guardar el tensor procesado
    torch.save(imagen, "firma_preprocesada.pt")

    print("Preprocesamiento de ejemplo terminado.")


if __name__ == "__main__":
    main()