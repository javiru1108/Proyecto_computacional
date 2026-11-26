from itertools import combinations, product
from pathlib import Path
import re
import torch
from torch.utils.data import random_split


def indexar_firmas(carpeta, prefijo): #aquí se va a retornar un diccionario con las firmas
    #prefijo se refiere a forgeries u original
    firmas = {}
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    patron = re.compile(rf"^{prefijo}_(\d+)_(\d+)$", re.IGNORECASE) #identifica el patrón "prefijo_número_número" que tienen todos los archivos 
    #el primer número es para el autor y el segundo para número de firma

    for ruta in carpeta.rglob("*"):
        if not ruta.is_file(): #se asegura de tomar archivos que sirvan (no directorios, o archivos corructos o que no existan)
            continue
        if ruta.suffix.lower() not in extensiones: #evita cualquier extensión que no nos sirva
            continue

        coincidencia = patron.match(ruta.stem) #toma el nombre del archivo sin extensión
        if coincidencia is None:
            continue

        escritor = int(coincidencia.group(1)) #el primer numero
        num_firma = int(coincidencia.group(2)) #segundo número

        if escritor not in firmas:
            firmas[escritor]={}

        if num_firma in firmas[escritor]:
            raise ValueError(f"Firma duplicada: {ruta}")

        firmas[escritor][num_firma] = ruta
    return firmas        

def generar_parejas(originales, falsas, escritores):
    pares = []

    for escritor in sorted(escritores): #recorre los escritores en orden
        firmas_org = originales[escritor]
        firmas_forg = falsas[escritor]
        ids = sorted(firmas_org.keys())

        # Comprobar que existen 24 firmas auténticas
        if len(ids) != 24:
            raise ValueError(
                f"El escritor {escritor} no tiene 24 firmas auténticas."
            )

        # Comprobar que existen las falsificaciones correspondientes
        if set(ids) != set(firmas_forg.keys()):
            raise ValueError(
                f"Las firmas del escritor {escritor} "
                "no coinciden entre originales y falsificaciones."
            )

        # Todas las parejas auténtica–auténtica, sin repetir ni comparar una firma consigo misma.
        for id_1, id_2 in combinations(ids, 2):
            img_1 = firmas_org[id_1]
            img_2 = firmas_org[id_2]

            pares.append((img_1, img_2, 0)) #Le pone etiqueta 0 a las firmas similares

        # Cada firma auténtica con cada falsificación del mismo autor.
        for id_org, id_forg in product(ids, sorted(firmas_forg)):
            img_1 = firmas_org[id_org]
            img_2 = firmas_forg[id_forg]

            pares.append((img_1, img_2, 1)) #Etiqueta 1 a firmas no similares

    return pares

def separar_datos():
    CARPETA_PROYECTO = Path(__file__).resolve().parent
    CARPETA_CEDAR = CARPETA_PROYECTO / "data" / "signatures"
    CARPETA_ORG = CARPETA_CEDAR / "full_org"
    CARPETA_FORG = CARPETA_CEDAR / "full_forg"

    for carpeta in (CARPETA_ORG, CARPETA_FORG):
        if not carpeta.is_dir():
            raise FileNotFoundError(
                f"No se encontró {carpeta}. Ejecuta descargar.sh para obtener los datos."
            )

    originales = indexar_firmas(CARPETA_ORG, "original")
    falsificaciones = indexar_firmas(CARPETA_FORG, "forgeries")

    escritores = sorted(set(originales.keys()) & set(falsificaciones.keys()))
    #escritores = [e for e in escritores if e != 41] #Arbitrariamente se excluye, en el artículo tmb se quitó este mismo pero no se xq
    escritores = [e for e in escritores if e <=10]

    escritores = list(escritores)

    num_train = int(0.7 * len(escritores)) #70% para entrenamiento y 30% para validación
    num_val = len(escritores) - num_train


    generator = torch.Generator().manual_seed(0)

    train, val = random_split(escritores, [num_train, num_val], generator=generator)

    escritores_train = list(train)
    escritores_val = list(val)

    parejas_train = generar_parejas(originales, falsificaciones, escritores_train)
    parejas_val = generar_parejas(originales, falsificaciones, escritores_val)

    return parejas_train, parejas_val

