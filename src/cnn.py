import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

train_folder = "./../images/training/"
test_folder = "./../images/ProcessedThresholdTest/"
validation_folder = "./../images/validation/"

epochs = 16
batch_size = 1

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
x_validation, y_validation = load_images(validation_folder)

#x_test /= 255.0

y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
y_validation_onehot = to_categorical(y_validation, num_classes=10)

#create a convolutional neural network model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(20, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation_onehot))

eval_loss, eval_acc = model.evaluate(x_test, y_test_onehot, batch_size=1)


predictions = model.predict(x_test)

predict = 0
for i in range(len(x_test)):
    true_label = y_test[i]
    predicted_label = np.argmax(predictions[i])
    print(f"Image {i+1}: Label réel: {true_label}, Prédiction: {predicted_label}, Confiance: {predictions[i][predicted_label]*100:.2f}%")
    if(true_label == predicted_label):
        predict += 1

print(f"Accuracy: {predict/len(x_test) * 100:.2f}%")

#extract in a text file the convultional filters and weights and biases
f = open("./poids_et_biais/convolutional_filters.txt", "w")
for layer in model.layers:
    if 'conv' in layer.name:
        weights, biases = layer.get_weights()
        f.write(f"Layer: {layer.name}\n")
        f.write(f"Weights: {weights}\n")
        f.write(f"Biases: {biases}\n")
f.close()

for layer in model.layers:
        array = layer.get_weights()
        if 'conv' in layer.name:   
            continue
        if len(array) != 0:
            w1,b1 = layer.get_weights()
            np.savetxt('poids_et_biais/cnn_weight_' + layer.name + '.txt', w1)
            np.savetxt('poids_et_biais/cnn_bias_' + layer.name + '.txt', b1)

# Print values for the flatten layer
#flatten_layer_output = model.get_layer('flatten').output
#flatten_layer_model = tf.keras.models.Model(inputs=model.input, outputs=flatten_layer_output)
#flatten_output = flatten_layer_model.predict(x_test[:1])
#print('Flatten layer output:' + str(flatten_output.shape))
#write the values in a text file
#np.savetxt('./poids_et_biais/flatten_layer_output.txt', flatten_output)

# Extract feature maps
#layer_outputs = [layer.output for layer in model.layers[:2]]
#activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#activations = activation_model.predict(x_test[:1])  # Getting the feature maps for the first test image

# Print values of the 8 feature maps
#for layer_activation in activations:
#    print(layer_activation.shape)  # Shape of the feature map tensor
#    for i in range(layer_activation.shape[-1]):
#        print(f"Feature Map {i+1}:")
#        print(layer_activation[0, :, :, i])  # Printing the values of the feature map
