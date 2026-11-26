import csv
from pathlib import Path
import matplotlib.pyplot as plt

ruta = Path(__file__).resolve().parent / "historial_entrenamiento.csv"

with open(ruta, "r") as archivo:
    lector = csv.DictReader(archivo)
    historial = list(lector)

epocas = [int(fila["epoca"]) for fila in historial]
loss_train = [float(fila["loss_train"]) for fila in historial]
loss_val = [float(fila["loss_val"]) for fila in historial]

plt.figure()

plt.loglog(epocas, loss_train, color="blue", marker="o", label="Entrenamiento")
plt.loglog(epocas, loss_val, color="red", marker="s", label="Validación")

plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Curvas de pérdida")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
