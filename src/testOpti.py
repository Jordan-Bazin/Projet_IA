import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

# Définition des chemins des dossiers contenant les images d'entraînement et de test
trainAugmented_folder = "./../images/trainingAugmented/"
train_folder = "./../images/training/"
trainTh_folder = "./../images/bmpProcessedThreshold/"
test_folder = "./../images/testGut/"
validation = "./../images/validation/"
trainTab = [trainTh_folder, train_folder, trainAugmented_folder]

#epochs = 22
#batch_size = 1

def load_images(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        image = tf.keras.preprocessing.image.load_img(folder_path+file, color_mode='grayscale', target_size=(28, 28 ))
        #image.convert("L")
        image = tf.keras.preprocessing.image.img_to_array(image)
        #image /= 255.0
        images.append(image)
        labels.append(int(file[0]))

    return np.array(images), np.array(labels)

x_train, y_train = load_images(train_folder)
x_test, y_test = load_images(test_folder)
x_validation, y_validation = load_images(validation)

# One-hot encodage des étiquettes
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
y_validation_onehot = to_categorical(y_validation, num_classes=10)


# Normalisation des images de test
x_test /= 255.0

def tryModel(epochs, mode):
    if(mode == 0):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())
        ])
    elif(mode == 1):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())
        ])
    elif(mode == 2):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())
        ])
    else:
        return -1

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=1)

    eval_loss, eval_acc = model.evaluate(x_test, y_test_onehot, batch_size=1)
    print('Accuracy: ', eval_acc*100)

    return eval_acc * 100

def showModels():
    epochs = 50
    nTry = 8 
    mode = 3 
    modelTab = [0] * mode

    for d in range(0, mode):
        AccuracyTab = [0] * epochs
        for i in range(1, epochs):
            accuracy = 0
            for j in range(0, nTry):
                accuracy += tryModel(i, d)

            AccuracyTab[i] = accuracy / nTry
        modelTab[d] = AccuracyTab
    print(modelTab)
   
    plt.plot(modelTab[0], label = "model 1")
    plt.plot(modelTab[1], label = "model 2")
    plt.plot(modelTab[2], label = "model 3")
    plt.legend()
    plt.show()  

def findMostMistakes():
    nTry = 50
    wrongPredictions = {}
    for i in range(0, nTry):
        wrongPredictions[i], = model()
    print(wrongPredictions)
    
def model(mode = "normal", epochs = 23, batch_size = 1, drop = 0.3, neurones = 256):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(neurones, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())
    ])
     
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_onehot), shuffle=True)

    # Évaluation du modèle avec les étiquettes one-hot
    eval_loss, eval_acc = model.evaluate(x_test, y_test_onehot, batch_size=1)
    print('Accuracy: ', eval_acc*100)

    # Prédiction sur les données de test
    predictions = model.predict(x_test)

    if(mode == "mostMistakes"): 
        wrongPredictions = []    
        for i in range(len(x_test)):
            true_label = y_test[i]
            predicted_label = np.argmax(predictions[i])
            print(f"Image {i+1}: Label réel: {true_label}, Prédiction: {predicted_label}")
            if(true_label != predicted_label):
                wrongPredictions.append(true_label)
        return wrongPredictions
    elif(mode == "normal"):
        accuracy = 0
        for i in range(len(x_test)):
            true_label = y_test[i]
            predicted_label = np.argmax(predictions[i])
            if(true_label == predicted_label):
                accuracy += 1
        return accuracy / len(x_test) * 100

def modelCnn(nMaps, mode = "normal", epochs = 23, batch_size = 1):
    modelCnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(nMaps, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(), input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())
    ])

    modelCnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    history = modelCnn.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_onehot))
    
    predictions = modelCnn.predict(x_test)

    accuracy = 0
    for i in range(len(x_test)):
        true_label = y_test[i]
        predicted_label = np.argmax(predictions[i])
        if(true_label == predicted_label):
            accuracy += 1
    return accuracy / len(x_test) * 100
    

def findBestNeurones():
    accuracyTmp = 0
    accuracyTab = {}
    accuracy = 0
    nTry = 15
    values128 = [0] * nTry 
    values256 = [0] * nTry
    values512 = [0] * nTry
    values1176 = [0] * nTry

    for i in range(0, nTry):
        accuracyTmp = model(neurones=128)
        values128[i] = accuracyTmp
        accuracy += accuracyTmp 
    accuracyTab["128"] = accuracy / nTry
    accuracyTmp = 0
    accuracy = 0

    for i in range(0, nTry):
        accuracyTmp = model(neurones=256)
        values256[i] = accuracyTmp
        accuracy += accuracyTmp 
    accuracyTab["256"] = accuracy / nTry
    accuracyTmp = 0
    accuracy = 0

    for i in range(0, nTry):
        accuracyTmp = model(neurones=512)
        values512[i] = accuracyTmp
        accuracy += accuracyTmp 
    accuracyTab["512"] = accuracy / nTry
    accuracyTmp = 0
    accuracy = 0

    for i in range(0, nTry):
        accuracyTmp = model(neurones=1.5 * 784)
        values1176[i] = accuracyTmp
        accuracy += accuracyTmp 
    accuracyTab["1176"] = accuracy / nTry
    accuracyTmp = 0
    accuracy = 0

    print(accuracyTab)

   
def testModel():
    nTry = 7
    epochs = 40
    avgAcc = [0] * epochs
    ecartsTypeAcc = [0] * epochs
    results = [[0] * nTry for _ in range(epochs)] 
    sumAcc = 0
    for i in range(0, epochs):
        for j in range(0, nTry):
            results[i][j] = model(epochs=i+1, neurones= 1.5 * 784, batch_size=1)

    for i in range(0, epochs):
        sumAcc = 0
        for j in range(0, nTry):
            sumAcc += results[i][j]
        avgAcc[i] = sumAcc / nTry
        ecartsTypeAcc[i] = np.std(results[i])

    print(avgAcc)
    print(ecartsTypeAcc)

    plt.plot(avgAcc, label = "average accuracy")
    plt.plot(ecartsTypeAcc , label = "standard deviation")
    plt.legend()
    plt.show()
    
def testBatchSize():
    nTry = 4
    epochs = 6
    batch_size = 2
    results = [0] * nTry

    for i in range(0, nTry):
        results[i] = model(epochs=epochs, batch_size=batch_size, neurones= 1.5 * 784)

    average = sum(results) / nTry
    std = np.std(results)

    print(average)
    print(std)

def testDataFolder():
    nTry = 10
    epochs = 50
    results = [[0] * epochs for _ in range(3)] 
    sum = 0
    for k in range(0, 3):
        train_folder = trainTab[k]
        x_train, y_train = load_images(train_folder)
        y_train_onehot = to_categorical(y_train, num_classes=10)
        for i in range(0, epochs):
            sum = 0
            for j in range(0, nTry):
                sum += model(x_train, y_train_onehot, epochs=i+1, neurones= 1.5 * 784, batch_size=1)
            results[k][i] = sum / nTry
       

    plt.plot(results[0], label = "Thresholded images")
    plt.plot(results[1], label = "Thresholded + processed images")
    plt.plot(results[2], label = "Augmented images")
    plt.legend()
    plt.show()

def testCnn():
    nTry = 8
    epochs = 40
    avgAcc = [0] * epochs
    nMaps = [8, 16, 24, 32]
    results = [[[0] * nTry for _ in range(epochs)] for _ in range(len(nMaps))]
    avgAcc = [[0] * epochs for _ in range(len(nMaps))]
    ecartsTypeAcc = [[0] * epochs for _ in range(len(nMaps))] 
    m = 0
    for featuremap in nMaps:
        for i in range(0, epochs):
            for j in range(0, nTry):
                results[m][i][j] += modelCnn(featuremap, epochs=i+1, batch_size=1)
            avgAcc[m][i] = sum(results[m][i]) / nTry 
            ecartsTypeAcc[m][i] = np.std(avgAcc[m][i])
        m += 1

    for i in range(0, len(nMaps)):
        plt.plot(avgAcc[i], label = f"{nMaps[i]} feature maps")
        #plt.plot(ecartsTypeAcc[i], label = f"{nMaps[i]} feature maps")
    plt.legend()
    plt.show()

#findBestNeurones() 
#findMostMistakes("findMistakes")
#showModels()
#testModel()
#testBatchSize()
#testDataFolder()
#testCnn()