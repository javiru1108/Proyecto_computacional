#!/usr/bin/env bash
set -euo pipefail

CARPETA_PROYECTO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CARPETA_DATA="$CARPETA_PROYECTO/data"
ARCHIVO="$CARPETA_DATA/signatures.rar"
LINK="https://cedar.buffalo.edu/NIJ/data/signatures.rar"

mkdir -p -- "$CARPETA_DATA"
if [ ! -s "$ARCHIVO" ]; then
    wget --timeout=30 --tries=2 -O "$ARCHIVO.part" "$LINK"
    mv -- "$ARCHIVO.part" "$ARCHIVO"
fi

unrar x -o- "$ARCHIVO" "$CARPETA_DATA/"
echo "Datos extraídos en $CARPETA_DATA"
