# Digits_Classifier
Machine learning for digits classification.   
Training done on the MNIST Digits dataset 8x8 pixels images.

## User Interface: **Digit Classifier App**
You can use the graphical user interface to draw your own digits and see the model's classification updating in real time.   

### Launch the app
Verify that you have python and the required libraries. Then, in the folder DigitClassifier_App enter the command   
```console
> python app.py
```
Enjoy the wondeful user experience and have fun drawing digits from 0 to 9 !   

### Comments and advices
The model has been trained on 8x8px images, you can see on the right how your drawing is toverted to this resolution. It helps to understand what the model "sees".   
I think that due to the low resolution training images, the model only takes into account the global picture and easily recognizable features. It makes the model at bit to hasty to classify some digits (the 9 for example with an easily recognisable "head").   
   
To get the best results with the 8x8px model:   
- Take all the vertical space available when you draw your digit
- Do **not** draw a bar on the bottom of the 1 (the small thing on which the number 1 "stands")   
   
To get the best results with the 28x28px model:   
- Don't use it, it is bad for now x)
   
I have to train a better model for the 28x28px images.   

## To Do
- Train a better model for dataset 28x28px images (More layers ? Convolutionnal layers ?)
- Try Dataset augmentation ?

## Current best model

### 8x8px Dataset
**SimpleNeuralNet**   

Neural network:   
input layer  -> 64 features   
hidden layer -> 32 neurons   
output layer -> 10 classes   

Activation function:   
ReLU   

Loss function:
Cross Entropy

Training:   
MNIST Digits dataset 8x8px (on 80% random training images)   
nb_epochs = 200000   
batch_size = 1024   
learning_rate = 0.00005   

Result:   
*96% accuracy* (on 20% random evaluation images from the dataset)   

### 28x28px Dataset
**DoubleLayerNeuralNet**   

Neural network:   
input layer  -> 784 features   
hidden layer 1 -> 128 neurons   
hidden layer 2 -> 128 neurons   
output layer -> 10 classes   

Activation function:   
ReLU   

Loss function:
Cross Entropy

Training:   
MNIST Digits train dataset 28x28px  
nb_epochs = 6000   
batch_size = 16384   
learning_rate = 0.0005   

Result:   
*97% accuracy* (on MNIST Digits test dataset 28x28px)   
