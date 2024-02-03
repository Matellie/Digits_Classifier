import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import models
import datasets

import matplotlib.pyplot as plt
import numpy as np
import time
import random

def set_cuda_device():
    if cuda.is_available():
        print(f'Using {cuda.get_device_name(0)}')
        DEVICE = torch.device('cuda:0')
    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')
    return DEVICE

def split_dataset(dataset, val_split, seed=42):
    indices = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices, val_indices

def plot_loss_graph(update_loss_graph, loss_history):
    plt.plot(
        [update_loss_graph*a for a in range(len(loss_history))],
        loss_history
    )
    plt.yscale('log')
    plt.title('Training loss')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

def show_images(inputs, labels, predictions):
    sqrt = int(np.ceil(np.sqrt(len(inputs))))
    print(f"Show {len(inputs)} images in a {sqrt}x{sqrt} grid")

    figure, axis = plt.subplots(sqrt, sqrt)
    for i in range(sqrt):
        for j in range(sqrt):
            if i*sqrt + j >= len(inputs):
                axis[i, j].axis('off')
            else:
                if labels[i*sqrt + j] == predictions[i*sqrt + j]:
                    color = 'green'
                else:
                    color = 'red'
                image = np.array(inputs[i*sqrt + j].detach().cpu(), dtype='float')
                pixels = image.reshape((28, 28))
                axis[i, j].imshow(pixels, cmap='gray')
                axis[i, j].set_title(f"L:{labels[i*sqrt + j].item()},P:{predictions[i*sqrt + j]}", fontsize=10, color=color)
                axis[i, j].axis('off')
    plt.subplots_adjust(hspace=1.2, wspace=1.2)
    plt.show()

def evaluate_model(model, val_loader, device, show_i=False):
    # Evaluate model
    with torch.no_grad():
        correct_guess = torch.tensor(0)
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            y_pred_class = y_pred.argmax(dim=1)
            correct_guess = correct_guess + y_pred_class.eq(labels).sum()

            if show_i:
                show_images(inputs, labels, y_pred_class)

        # Compute and print accuracy
        accuracy = correct_guess / float(len(val_loader.sampler))
        
        return accuracy, correct_guess

def main():
    device = set_cuda_device()
    batch_size = 16384
    nb_workers = 0
    learning_rate = 0.0005
    nb_epochs = 200000
    update_print = 10000
    update_loss_graph = 100
    update_best_model = 1000

    # Load dataset
    dataset_train = datasets.MNISTDigits28x28_train()
    dataset_test = datasets.MNISTDigits28x28_test()
    
    print(f"Train dataset: {len(dataset_train)}, Test dataset: {len(dataset_test)}")
    print(f"Dataset: {dataset_train.nb_features} features, {dataset_train.nb_classes} classes")
    
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=nb_workers, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=nb_workers, shuffle=True)

    # Create model
    model = models.DoubleLayerNeuralNet(input_size=dataset_train.nb_features, hidden_size1=128, hidden_size2=128, nb_classes=dataset_train.nb_classes)
    model = model.to(device)

    # Set loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set best model and best loss
    best_model = model
    best_accuracy = 0

    # Train model
    loss_history = []
    time_train, time_epoch = time.time(), time.time()
    plt.ion()
    print(f'{nb_epochs} epochs, print update each {update_print} epochs')
    for epoch in range(nb_epochs):
        loss_epoch = 0

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            l = loss(y_pred, labels)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += l.item()

        # Reporting
        if (epoch+1) % update_print == 0:
            # Print epoch training time and some infos
            print(f'Epoch {epoch+1}: loss = {loss_epoch:.4f}, time = {(time.time() - time_epoch):.1f}s')
            time_epoch = time.time()

        if (epoch+1) % update_loss_graph == 0:
            #Save loss history
            loss_history.append(loss_epoch)
            # Plot loss history graph
            plot_loss_graph(update_loss_graph, loss_history)

        if (epoch+1) % update_best_model == 0:
            # Evaluate model
            accuracy, correct_guess = evaluate_model(model, test_loader, device)
            # Save best model
            if accuracy > best_accuracy:
                print(f"New best model: {accuracy * 100:.1f}% accuracy ({correct_guess}/{len(test_loader.sampler)})")
                best_accuracy = accuracy
                best_model = model
    print(f'Training time: {(time.time() - time_train):.1f}s')
    plt.ioff()

    # Evaluate best model
    accuracy, correct_guess = evaluate_model(best_model, test_loader, device, show_i=False)
    print(f"Correct guess: {correct_guess}/{len(test_loader.sampler)}")
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save model and loss graph
    model_id = f'{dataset_train.name}_{best_model.name}_{accuracy * 100:.0f}'
    model_name =        'model_' +  model_id + '.pt'
    loss_graph_name =   'loss_' +   model_id + '.png'
    save_path = '.'
    models.save_model(best_model, save_path=save_path, model_name=model_name)
    models.save_loss_graph(loss_history, save_path=save_path, graph_name=loss_graph_name)

if __name__ == '__main__':
    main()