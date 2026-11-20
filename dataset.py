import torch 
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocesar_imagen
from preprocessing import calcular_desviacion_estandar

class SignatureDataset(Dataset):
    def __init__(self, pares, desviacion_estandar):
        self.pares = pares
        self.desviacion_estandar = desviacion_estandar

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, index):
        ruta_1, ruta_2, etiqueta = self.pares[index]
        img_1 = preprocesar_imagen(ruta_1, self.desviacion_estandar)
        img_2 = preprocesar_imagen(ruta_2, self.desviacion_estandar) #estas dos devuelven tensores de dimensiones (155, 220)

        img_1 = img_1.unsqueeze(0) #para la arquitectura se ocupa en nn.Conv2d una dimensión para el canal, con esto se agrega
        img_2 = img_2.unsqueeze(0)

        etiqueta = torch.tensor(etiqueta, dtype=torch.float32)

        return img_1, img_2, etiqueta


def datasets(parejas_train, parejas_val):
    imagenes_train = set()
    for ruta_1, ruta_2, etiqueta in parejas_train:
        imagenes_train.add(ruta_1)
        imagenes_train.add(ruta_2)

    desviacion = calcular_desviacion_estandar(sorted(imagenes_train))

    train_dataset = SignatureDataset(parejas_train, desviacion)
    val_dataset = SignatureDataset(parejas_val, desviacion)

    return train_dataset, val_dataset

def dataloader(train_dataset, val_dataset):
    BATCH_SIZE = 8 #la idea es hacerlo con 32, pero para probar intento con uno mas peque
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader 

