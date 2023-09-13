import os
import cv2

# Especifica a pasta que contém os arquivos de imagem
image_folder = "./images/apples"

# Cria um array para armazenar as imagens
images = []

# Percorre os arquivos na pasta
for file in os.listdir(image_folder):
    # Abre o arquivo de imagem
    image = cv2.imread(os.path.join(image_folder, file))

    # Adiciona a imagem ao array
    images.append(image)

    cv2.imshow("Image", image)

    # Aguarda o usuário pressionar uma tecla
    cv2.waitKey(0)


import numpy as np

# Cria um array de números inteiros
array = np.ones(10, dtype=int)*0 

# Imprime o array
print(array)
