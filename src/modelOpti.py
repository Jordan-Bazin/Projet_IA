import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from keras.models import Model

from tensorflow.keras.utils import to_categorical

# Définition des chemins des dossiers contenant les images d'entraînement et de test
train_folder = "./../images/training/"
test_folder = "./../images/testGut/"
validation_folder = "./../images/validation/"
singleTest_folder = "./../images/singleTest/"

epochs = 23
batch_size = 1

def load_images(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        image = tf.keras.preprocessing.image.load_img(folder_path+file, color_mode='grayscale', target_size=(28, 28 ))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image /= 255.0 # Normalisation des valeurs de pixels
        images.append(image)
        labels.append(int(file[0]))

    return np.array(images), np.array(labels)

x_train, y_train = load_images(train_folder)
x_test, y_test = load_images(test_folder)
x_validation, y_validation = load_images(validation_folder)

x_singleTest, y_singleTest = load_images(singleTest_folder)

# One-hot encodage des étiquettes
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
y_validation_onehot = to_categorical(y_validation, num_classes=10)

# Création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle avec categorical_crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle avec les étiquettes one-hot
history = model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_onehot))

# Évaluation du modèle avec les étiquettes one-hot
eval_loss, eval_acc = model.evaluate(x_test, y_test_onehot, batch_size=1)
print('Accuracy: ', eval_acc*100)

# Prédiction sur les données de test
predictions = model.predict(x_test)

# Affichage des prédictions dans le terminal
predict = 0
for i in range(len(x_test)):
    true_label = y_test[i]
    predicted_label = np.argmax(predictions[i])
    print(f"Image {i+1}: Label réel: {true_label}, Prédiction: {predicted_label}, Confiance: {predictions[i][predicted_label]*100:.2f}%")
    if(true_label == predicted_label):
        predict += 1

print(f"Accuracy: {predict/len(x_test) * 100:.2f}%")

print(predict)
    
def extractLayers(model):
    output_dir = 'poids_et_biais'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for layer in model.layers:
        array = layer.get_weights()
        if len(array) != 0:
            w1,b1 = layer.get_weights()
            np.savetxt('layers/layer_weight_' + layer.name + '.txt', w1)
            np.savetxt('layers/layer_bias_' + layer.name + '.txt', b1)

extractLayers(model)