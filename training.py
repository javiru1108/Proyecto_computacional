import torch
import torch.nn.functional as F




def contrastative_loss(distancia, etiquetas, margen=1.0): #usamos 1 para firmas falsificadas y 0 para similares
    etiquetas = etiquetas.float().view(-1)
    distancia = distancia.view(-1) #Los view son para asegurar que sean de una dimensión
    loss = ((1-etiquetas) * distancia.pow(2) + etiquetas * F.relu(margen - distancia).pow(2) )

    return loss.mean()
    

def ejecutar_epoca(model, dataloader, device, optimizer=None, margin=1.0):

    entrenamiento = optimizer is not None

    if entrenamiento: #Comprueba si hay que entrenar o evaluar el asunto
        model.train()
    else:
        model.eval()
    loss_tot = 0.0
    aciertos=0
    total=0

    for img1, img2, etiquetas in dataloader:
        #pasa las varas al dispositivo
        img1=img1.to(device)
        img2=img2.to(device)
        etiquetas=etiquetas.to(device, dtype=torch.float32).view(-1) #colapsado a unidimensional

        #cero a gradientes
        if entrenamiento:
            optimizer.zero_grad()

        # forward propagation
        with torch.set_grad_enabled(entrenamiento):
            distancias = model(img1, img2)
            loss= contrastative_loss(distancias, etiquetas, margin)

        #backpropagation y la actualización
        if entrenamiento:
            loss.backward()
            optimizer.step()

        #acumular el loss
        batch_size=etiquetas.size(0)
        loss_tot += loss.item() * batch_size

        #Ver exactitud
        predicciones = (distancias.detach() >= 0.5).long()

        aciertos += (predicciones == etiquetas.long()).sum().item()

        total += batch_size

    if total == 0:
        raise ValueError("No hay batches completos en el Dataloader")
    
    loss_promedio = loss_tot / total
    exactitud = aciertos / total

    return loss_promedio, exactitud