from .model import LinearRegression, SimpleNeuralNet

from torch import save, load
import os

def save_model(model, save_path, model_name='model.pt'):
    save(model.state_dict(), os.path.join(save_path, model_name))

def load_model(model, model_path, model_name='model.pt'):
    model.load_state_dict(load(os.path.join(model_path, model_name)))
    return model