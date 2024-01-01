# Digits_Classifier
Machine learning for digits classification.   
Training done on the MNIST Digits dataset 8x8 pixels images.

## Current model
Neural network:   
input layer  -> 64 features   
hidden layer -> 32 neurons   
output layer -> 10 classes   

Training:   
MNIST Digits dataset (on 80% random training images)   
nb_epochs = 200000   
batch_size = 1024   
learning_rate = 0.00001   

Results:   
*97% accuracy* (on 20% random evaluation images from the dataset)   

## To Do
- User interface to write digits
- Train for bigger images -> Dataset augmentation
