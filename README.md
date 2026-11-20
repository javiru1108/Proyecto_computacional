# Siamese_Network_Pytorch
Proyecto final de física computacional y parte del proyecto de métricas para las firmas del pintor Fausto Pacheco

INSTRUCCIONES DE INSTALACIÓN DE DATOS

Bienvenidos a nuestro proyecto final del curso de física computacional. Este consiste en utilizar una métrica de diferenciación de imágenes propuesta en el paper "SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification" de Dey et al, para diferenciar firmas falsificadas de firmas reales. Para ello se utilizará la base de datos de firmas CEDAR, la cual ya incluye una clasificación de firmas reales vs. falsificadas. Para correr este programa será necesario que usted, el usuario, descargue estos datos en su operador. Para ello incluimos descargar.sh y las instrucciones a continuación:

1. Se recomienda abrir una carpeta en el ordenador donde se puede clonar el repositorio.
2. Copiar el enlace SSH del repositorio.
3. Dentro de la carpeta, correr git clone (link al repo).
4. Verificar que tienen acceso al repositorio en su GitHub.
5. En la terminal, correr: bash descargar.sh
6. A continuación, verificar que exista una carpeta de datos y dentro de esta las firmas en formato png.

INSTALAR PYTORCH 
El documento signet.sh tiene las bibliotecas que se utilizarán en el (preprocesamiento) proyecto. Si ya tiene pytorch en su computadora no hace falta instalarlo. 

PREPROCESAMIENTO DE DATOS 

Para mantener la concordancia con el paper propuesto se hará el mismo preprocesamiento sobre nuestras imágenes. En este se hacen los siguientes pasos: 

0. Lectura de los píxeles de las imágenes.
1. Utilizando interpolación bilineal se ajusta el tamaño de todas las imágenes a 155 × 220.
2. Se invierten las imágenes tal que los píxeles del fondo tengan un valor de 0.
3. Se normalizan las imágenes según la desviación estándar de los píxeles en el set de datos. 
4. Comprobación e impresión de pares Falso-Verdadero, Falso-Falso y Verdadero-Verdadero.

ARQUITECTURA

0. 

