import torch
import torch.nn as nn


class SigNet(nn.Module):

    def __init__(self):
        super().__init__()

        # 1 Primera capa convolucional 11x11 (96) + ReLU
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=96,
            kernel_size=11,
            stride=1,
            padding=0
        )

        self.relu = nn.ReLU()

        # 2 Normalización local + MaxPooling
        self.normalizacion = nn.LocalResponseNorm(
            size=5,
            alpha=1e-4,
            beta=0.75,
            k=2
        )

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        # 3 Segunda capa convolucional 5x5 (256) + ReLU
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # 4 Segunda normalización + MaxPooling
        self.normalizacion2 = nn.LocalResponseNorm(
            size=5,
            alpha=1e-4,
            beta=0.75,
            k=2
        )

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        # 5 Tercera capa convolucional 3x3 (384) + ReLU
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 6 Cuarta capa convolucional 3x3 (256) + ReLU
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 7 Tercer MaxPooling + Dropout
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        self.dropout3 = nn.Dropout(p=0.3)

        # 8 Primera capa completamente conectada
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=256 * 17 * 25,
            out_features=1024
        )

        self.dropout4 = nn.Dropout(p=0.5)

        # 9 Segunda capa completamente conectada
        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=128
        )


    def forward(self, x):

        # Adaptar las dimensiones de la imagen
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.ndim == 3:
            x = x.unsqueeze(0)

        # Primera convolución
        x = self.conv1(x)
        x = self.relu(x)

        # Primera normalización y MaxPooling
        x = self.normalizacion(x)
        x = self.maxpool1(x)

        # Segunda convolución
        x = self.conv2(x)
        x = self.relu(x)

        # Segunda normalización y MaxPooling
        x = self.normalizacion2(x)
        x = self.maxpool2(x)

        # Tercera convolución
        x = self.conv3(x)
        x = self.relu(x)

        # Cuarta convolución
        x = self.conv4(x)
        x = self.relu(x)

        # Tercer MaxPooling y Dropout
        x = self.maxpool3(x)
        x = self.dropout3(x)

        # Primera capa completamente conectada
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # Vector de 128 características
        x = self.fc2(x)

        return x

#tmb hay que definir una clase que use esta arquitectura para ya hacer una red siamesa
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SigNet() #se referencia a SigNet, encoder se vuelve un submódulo de la clase
        #Además este objeto se usa para ambas imagenes, entonces no se crea una nueva instancia, por lo que se comparten todos los pesos y sesgos, la def de una red siamesa
    def forward(self, img1, img2):
        encoding_1= self.encoder(img1)
        encoding_2= self.encoder(img2)

        #distancia euclidiana
        dist_cuadrada = torch.sum((encoding_1-encoding_2)**2, dim=1)

        distancia = torch.sqrt(torch.clamp(dist_cuadrada, min=1e-7)) #aquí el clamp lo que hace es evitar tener valores menores que cero, 
        #porque ya no hay que acercarlos más
        return distancia