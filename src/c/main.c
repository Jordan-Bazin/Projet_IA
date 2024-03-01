/*******************************************************
Nom ......... : main.c
Role ........ : Programme principal executant la lecture
                d'une image bitmap
Auteur ...... : Frédéric CHATRIE
Version ..... : V1.1 du 1/2/2021
Licence ..... : /

Compilation :
make veryclean
make
Pour exécuter, tapez : ./all
********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Bmp2Matrix.h"

float relu(float x)
{
    if (x < 0)
    {
        return 0;
    }
    return x;
}

void softmax(float *input, int length) {
    float m = -INFINITY;
  for (size_t i = 0; i < length; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < length; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < length; i++) {
    input[i] = expf(input[i] - offset);
  }
}

void softmaxT(float *input, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += exp(input[i]);
    }
   
    for(int i = 0; i < length; i++)
    {
        input[i] = exp(input[i]) / sum;
    }
}

int countLines(FILE *pFichier)
{
    char c;
    int nbLines = 0;
    while (!feof(pFichier))
    {
        c = fgetc(pFichier);
        if (c == '\n')
        {
            nbLines++;
        }
    }
    rewind(pFichier);
    return nbLines;
}

void flatten(unsigned char **mPixelsGray, float *flattenPixels)
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            flattenPixels[i * 28 + j] = (float)mPixelsGray[i][j];
        }
    }
}

void flatten3d(float ***input, float *flattenPixels, int nFilters)
{
    for(int i = 0; i < 26; i++)
    {
        for(int j = 0; j < 26; j++)
        {
            for(int k = 0; k < nFilters; k++)
            {
                flattenPixels[nFilters * ((26 *i) + j) + k] = input[k][i][j];
            }
        }
    }
}

void dense(float *layer, float *flattenPixels, float *weights, float *bias, int weightsSize, int biasSize)
{
    for(int i = 0; i < biasSize; i++)
    {
        layer[i] = 0;
    }

    for (int i = 0; i < weightsSize; i++)
    {   
        for (int j = 0; j < biasSize; j++)
        {
            layer[j] += flattenPixels[i] * weights[i * biasSize + j];
        } 
    }
    for(int i = 0; i < biasSize; i++)
    {
        layer[i] += bias[i];
    }

    if(biasSize == 10)
    {       
        softmax(layer, biasSize); 
    }
    else 
    {
        for (int i = 0; i < biasSize; i++)
        {
            layer[i] = relu(layer[i]);
        }
    }
}

void conv2D(float **image, float ***filters, float ***convOutput, float *bias, int nFilters)
{
    for(int i = 0; i < nFilters; i++)
    {
        for(int j = 0; j < 26; j++)
        {
            for(int k = 0; k < 26; k++)
            {
                for(int l = 0; l < 3; l++)
                {
                    for(int m = 0; m < 3; m++)
                    {
                        convOutput[i][j][k] += image[j + l][k + m] * filters[i][l][m];
                    }
                }
                convOutput[i][j][k] += bias[i];
                convOutput[i][j][k] = relu(convOutput[i][j][k]);
            }
        }
    }
}

void getWeights(float *weights, FILE *pfichierWeights, int size, int depth)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < depth; j++) {
            if (fscanf(pfichierWeights, "%f", &weights[i * depth + j]) != 1) {
                printf("Erreur lors de la lecture des données.\n");
            }
        }
    }
}

void getBiases(float *bias, FILE *pfichierBiases, int size)
{
    char *line = (char *)malloc(size * sizeof(char));
    int i = 0;
   
    while (fgets(line, 20, pfichierBiases) != NULL && i < size)
    {
        bias[i] = atof(line);
        i++;
    }
    free(line);
}

void getCnnWeights(float ***filtersWeights, int nFilters)
{
    char weightsPath[50]; 

    int j = 0;
    int k = 0;
    for(int i = 0;i < 9; i++)
    {
        sprintf(weightsPath, "./../poids_et_biais/filters/%d.txt", i);
        FILE *pfichierWeightsCnn = fopen(weightsPath, "rb");
        if(i >= 3 && i < 6)
        {
            k = 1;
            j = i - k - 2;
        }
        else if(i >= 6)
        {
            k = 2;
            j = i - k*2 - 2;
        }
        for(int y = 0; y < nFilters; y++) 
        {
            if (fscanf(pfichierWeightsCnn, "%f", &filtersWeights[y][k][j]) != 1)
            {
                printf("Erreur lors de la lecture des données.\n");
            }
        }
        if(i < 3)
        {
            j++;
        }
        fclose(pfichierWeightsCnn);
    }
}

void getCnnBiases(float *bias, int nFilters)
{
    char biasesPath[50];
    sprintf(biasesPath, "./../poids_et_biais/filters/filters_biases.txt");
    FILE *pfichierBiasesCnn = fopen(biasesPath, "rb");
    
    for(int i = 0;i < nFilters; i++)
    {
        if (fscanf(pfichierBiasesCnn, "%f", &bias[i]) != 1)
        {
            printf("Erreur lors de la lecture des données.\n");
        }
    }
}

int predict(float *layer)
{
    int max = 0;
    float confidence = 0;
    for (int i = 0; i < 10; i++)
    {
        printf("predictions : %f\n",layer[i]);
        if (layer[i] > layer[max])
        {
            max = i;
            confidence = layer[i] * 100;
        }
    }
    printf("Prediction : %d\t confidence : %f\n", max, confidence);
    return max;
}

int sequence(unsigned char **mPixelsGray)
{
    int i = 1;
    int prediction = 0;

    float *flattenPixels = (float *)malloc(28 * 28 * sizeof(float));
    flatten(mPixelsGray, flattenPixels);

    while (i != 0)
    {
        char weightsPath[100];
        char biasesPath[100];

        sprintf(weightsPath, "./../poids_et_biais/weights_%d.txt", i);
        FILE *pfichierWeights = fopen(weightsPath, "rb");
        if (pfichierWeights == NULL)
        {
            printf("Lecture terminée\n");
            prediction = predict(flattenPixels);
            break;
        }

        sprintf(biasesPath, "./../poids_et_biais/biases_%d.txt", i);
        FILE *pfichierBiases = fopen(biasesPath, "rb");
        if (pfichierBiases == NULL)
        {
            printf("Lecture terminée\n");
            break;
        }
        
        int sizeBias = countLines(pfichierBiases);
        int sizeWeights = countLines(pfichierWeights);

        float *bias = (float *)malloc(sizeBias * sizeof(float));
        getBiases(bias, pfichierBiases, sizeBias);

        float *weights = (float *)malloc(sizeWeights * sizeBias * sizeof(float *));
        getWeights(weights, pfichierWeights, sizeWeights, sizeBias);

        fclose(pfichierWeights);
        fclose(pfichierBiases);

        float *layer = (float *)malloc(sizeBias * sizeof(float)); // vecteur temporaire pour stocker les valeurs d'entrées de la nouvelle couche

        dense(layer, flattenPixels, weights, bias, sizeWeights, sizeBias);

        free(flattenPixels); // On free le vecteur d'entrée pour l'adapter a la prochaine couche
        flattenPixels = (float *)malloc(sizeBias * sizeof(float));
        
        for(int i = 0; i < sizeBias; i++)
        {
            flattenPixels[i] = layer[i];
        }
        printf("\n");
        free(bias);
        free(weights);
        free(layer);

        i++;
    }
    free(flattenPixels);
    return prediction;
}

int sequenceCnn(unsigned char **mPixelsGray, int nFilters) 
{
    int filterSize = 3;
    int prediction = 0;
    float *flattenPixels = (float *)malloc(28 * 28 * sizeof(float));
    flatten(mPixelsGray, flattenPixels);

    float ***filtersWeights = (float ***)malloc(nFilters * sizeof(float**)); 
    for (int i = 0; i < nFilters; i++)
    {
        filtersWeights[i] = (float **)malloc(filterSize * sizeof(float*));
        for (int j = 0; j < 3; j++)
        {
            filtersWeights[i][j] = (float *)malloc(filterSize * sizeof(float));
        }
    }
    getCnnWeights(filtersWeights, nFilters);

    float *bias = (float *)malloc(nFilters * sizeof(float));
    getCnnBiases(bias, nFilters);

    float ***convOutput = (float ***)malloc(nFilters * sizeof(float**));
    for (int i = 0; i < nFilters; i++)
    {
        convOutput[i] = (float **)malloc(26 * sizeof(float*));
        for (int j = 0; j < 26; j++)
        {
            convOutput[i][j] = (float *)malloc(26 * sizeof(float));
        }
    }

    float **image = (float **)malloc(28 * sizeof(float*)); // matrice qui cast l'image en float
    for (int i = 0; i < 28; i++)
    {
        image[i] = (float *)malloc(28 * sizeof(float));
    }
    for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j < 28; j++)
        {
            image[i][j] = (float)mPixelsGray[i][j];
        }
    }

    conv2D(image, filtersWeights, convOutput, bias, nFilters);

    float *flattenConvOutput = (float *)malloc(26 * 26 * nFilters * sizeof(float));

    flatten3d(convOutput, flattenConvOutput, nFilters);

    FILE *pfichierWeights = fopen("./../poids_et_biais/cnn_weight_dense.txt", "rb");
    FILE *pfichierBiases = fopen("./../poids_et_biais/cnn_bias_dense.txt", "rb");

    int sizeBias = countLines(pfichierBiases);
    int sizeWeights = countLines(pfichierWeights);

    float *biasDense = (float *)malloc(sizeBias * sizeof(float));
    getBiases(biasDense, pfichierBiases, sizeBias);

    float *weightsDense = (float *)malloc(sizeWeights * sizeBias * sizeof(float *));
    getWeights(weightsDense, pfichierWeights, sizeWeights, sizeBias);

    fclose(pfichierWeights);
    fclose(pfichierBiases);

    float *output = (float *)malloc(sizeBias * sizeof(float));

    dense(output, flattenConvOutput, weightsDense, biasDense, sizeWeights, sizeBias);

    prediction = predict(output);

    free(flattenPixels); 
    free(flattenConvOutput);
    free(bias);
    free(biasDense);
    free(weightsDense);
    free(output);
    free(filtersWeights);
    free(convOutput);

    return prediction;

}

void simpleTest(int num) {
    BMP bitmap;
    FILE *pFichier = NULL;
    char filePath[20];

    sprintf(filePath, "image-test/%d_0.bmp", num);
    pFichier = fopen(filePath, "rb"); // Ouverture du fichier contenant l'image

    if (pFichier == NULL)
    {
        printf("%s\n", "0_0.bmp");
        printf("Erreur dans la lecture du fichier\n");
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier); // Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);

    sequence(bitmap.mPixelsGray);

    DesallouerBMP(&bitmap);

}

void cnnTest(int num, int j, int nFilters) {
    BMP bitmap;
    FILE *pFichier = NULL;
    int prediction = 0;
    char filePath[20];

    sprintf(filePath, "image-test/%d_%d.bmp", num, j);
    pFichier = fopen(filePath, "rb"); // Ouverture du fichier contenant l'image

    if (pFichier == NULL)
    {
        printf("%s\n", "0_0.bmp");
        printf("Erreur dans la lecture du fichier\n");
    }
    LireBitmap(pFichier, &bitmap);
    fclose(pFichier); // Fermeture du fichier contenant l'image

    ConvertRGB2Gray(&bitmap);

    prediction = sequenceCnn(bitmap.mPixelsGray, nFilters);
    printf("Label réel : %d, Prediction : %d\n",num, prediction);

    DesallouerBMP(&bitmap);

}

int main(int argc, char *argv[])
{
    //simpleTest(3);
    cnnTest(2, 0, 20); // last arg is for the number of filters
    return 0;
}
