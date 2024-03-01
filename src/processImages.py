import os
import cv2

# Chemin du dossier contenant les images d'entrée et de sortie
input_folder = "./../images/training"
output_folder = "./../images/trainingAugmented"

# Création du dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Liste de fichiers dans le dossier d'entrée
files = os.listdir(input_folder)

i = 310

# Boucle à travers tous les fichiers dans le dossier
for file in files:
    # Chemin complet du fichier d'entrée
    input_path = os.path.join(input_folder, file)
    
    # Lecture de l'image
    image = cv2.imread(input_path)
    
    # Récupération des dimensions de l'image
    height, width = image.shape[:2]
    
    # Création de la matrice de rotation
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1)
    
    # Rotation de l'image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Ajout de "d10" à la fin du nom de fichier avant l'extension
    filename, extension = os.path.splitext(file)
    output_filename = filename[0] + "___" + str(i) + extension
    i += 1

    # Chemin complet du fichier de sortie
    output_path = os.path.join(output_folder, output_filename)
    
    # Enregistrement de l'image tournée
    cv2.imwrite(output_path, rotated_image)

print("Rotation des images terminée.")
