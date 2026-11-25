from pathlib import Path
import torch 
import csv
from crear_pares import separar_datos
from dataset import datasets, dataloader
from arqui import SiameseNetwork
from training import ejecutar_epoca
def main():
    #en principio esas son todas las cosas que necesitamos importar

    SEMILLA = 0
    EPOCHS = 20
    MARGEN = 1.0

    torch.manual_seed(SEMILLA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #la idea de esto es poder usar la gpu si es posible

    CARPETA_PROYECTO = Path(__file__).resolve().parent
    CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"

    CARPETA_RESULTADOS.mkdir(exist_ok=True) 
    # esto hace el directorio si no existía antes, pero no tira error si ya existe

    RUTA_MODELO = CARPETA_RESULTADOS / "mejor_modelo.pth"
    RUTA_HISTORIAL = CARPETA_RESULTADOS / "historial_entrenamiento.csv"

    #en principio ya está lista toda la configuración que se necesitaba

    parejas_train, parejas_val = separar_datos()

    train_datset, val_dataset = datasets(parejas_train, parejas_val)

    train_loader, val_loader = dataloader(train_datset, val_dataset)

    #Ya con esto quedan las parejas, los datasets y dataloaders listos
    #ahora toca inicializar red siamesa y optimizador

    model = SiameseNetwork().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-7)


    # ciclo para entrenamiento y validación

    mejor_loss_val = float("inf") #así se va actualizando siempre que sea menor
    historial=[]

    for epoca in range(EPOCHS):
        loss_train, exac_train = ejecutar_epoca(model, train_loader, device, optimizer=optimizer, margin=MARGEN)

        loss_val, exac_val = ejecutar_epoca(model, val_loader, device, optimizer=None, margin=MARGEN)

        historial.append({"epoca": epoca+1, "loss_train": loss_train, "exactitud_train": exac_train, "loss_val": loss_val, "exactitud_val": exac_val})

        #hay que guardar el modelo con menor loss
        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            torch.save({
                "epoca": epoca + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_val": loss_val,
                "accuracy_val": exac_val,
                "desviacion_estandar":
                    train_datset.desviacion_estandar,
                "margen": MARGEN,
                "umbral": 0.5
            }, RUTA_MODELO)

    #ahora hay que guardar el historial de training
    with open(RUTA_HISTORIAL, "w", newline="", encoding="utf-8") as archivo:
        campos=["epoca", "loss_train", "exactitud_train", "loss_val", "exactitud_val"]
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(historial)
    print("mejor loss: ", mejor_loss_val)
    print("modelo guardado en ", RUTA_MODELO)
    print("historial guardado en ", RUTA_HISTORIAL)

if __name__=="__main__":
    main()
