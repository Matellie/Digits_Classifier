from models import *

from tkinter import *
from PIL import ImageGrab, Image, ImageTk
import torch
import numpy as np

class ClassificationModels:
    def __init__(self):
        # Load models
        model_8x8 = SimpleNeuralNet(input_size=64, hidden_size=32, nb_classes=10)
        self.model_8x8 = load_model(model=model_8x8, model_path='model_ckpt', model_name='model_MNISTDigits8x8_SimpleNeuralNet_96.pt')

        model_28x28 = DoubleLayerNeuralNet(input_size=784, hidden_size1=128, hidden_size2=128, nb_classes=10)
        self.model_28x28 = load_model(model=model_28x28, model_path='model_ckpt', model_name='model_MNISTDigits28x28_DoubleLayerNeuralNet_97.pt')

        # Set default model
        self.model_pred = self.model_8x8
        self.img_size = 8

    def switch_model(self):
        if self.model_pred == self.model_8x8:
            self.model_pred = self.model_28x28
            self.img_size = 28
        else:
            self.model_pred = self.model_8x8
            self.img_size = 8

def main():

    def draw(event):
        # Draw on canvas
        x, y = event.x, event.y
        if canvas.old_coords:
            x1, y1 = canvas.old_coords
            canvas.create_line(x, y, x1, y1, width=50, fill='white')
        canvas.old_coords = x, y

        # Get image and display it
        image_process, image_display = get_images()
        display_image(image_display)

        # Predict digit and display it
        prediction = predict(image_process)
        display_prediction(prediction)

    def reset_coords(event):
        canvas.old_coords = None

    def get_images():
        x1 = app.winfo_rootx() + canvas.winfo_x() + left_frame.winfo_x()
        y1 = app.winfo_rooty() + canvas.winfo_y() + left_frame.winfo_y()
        x2 = x1 + canvas.winfo_width()
        y2 = y1 + canvas.winfo_height()
        image = ImageGrab.grab().crop((x1+5, y1+5, x2-5, y2-5))

        image_process = image.resize((models.img_size, models.img_size))
        image_display = image_process.resize((150, 150), Image.NEAREST)

        return image_process, image_display
    
    def display_image(image):
        image = ImageTk.PhotoImage(image)
        img_panel.configure(image=image)
        img_panel.image = image

    def predict(image):
        pred = None
        with torch.no_grad():
            image = np.array(image).astype(np.float32)[:,:,0]
            image = image.flatten()
            image = torch.from_numpy(image)

            pred = models.model_pred(image)
            pred = torch.argmax(pred).item()
        
        return pred
    
    def display_prediction(pred):
        pred_area.configure(text=str(pred))

    def change_model():
        models.switch_model()
        model_used.configure(text="Model used : " + str(models.img_size) + "x" + str(models.img_size))

    # Load models
    models = ClassificationModels()

    # Main window
    app = Tk()
    app.geometry("800x550")
    app.title("Digit Detector")
    app.configure(bg='white')

    # App title
    app_title = Label(app, text="Digit Detector\n", font=("Arial", 20), bg='white')
    app_title.pack()

    # Left frame (draw area and clear button)
    left_frame = Frame(app, bg='white')
    left_frame.pack(fill=BOTH, side=LEFT)

    draw_area_title = Label(left_frame, text="Draw a digit below", font=("Arial", 15), bg='white')
    draw_area_title.pack()

    canvas = Canvas(left_frame, width=400, height=400)
    canvas.configure(bg='black')
    canvas.pack()
    canvas.old_coords = None

    clear_button = Button(left_frame, text="Clear", font=("Arial", 15), width=20, command=lambda: canvas.delete("all"))
    clear_button.pack(side=BOTTOM)

    # Right frame (image 8x8px and model prediction)
    right_frame = Frame(app, bg='white')
    right_frame.pack(fill=BOTH, expand=True)

    img_display_title = Label(right_frame, text="Image used by the model", font=("Arial", 15), bg='white')
    img_display_title.pack()

    img = ImageTk.PhotoImage(Image.new('RGB', (150, 150), color='black'))
    img_panel = Label(right_frame, image=img)
    img_panel.image = img
    img_panel.pack()

    pred_area_title = Label(right_frame, text="\nPrediction", font=("Arial", 15), bg='white')
    pred_area_title.pack()

    pred_area = Label(right_frame, text="None", font=("Arial", 20), bg='white')
    pred_area.pack()

    change_model_button = Button(right_frame, text="Change model", font=("Arial", 15), command=change_model)
    change_model_button.pack(side=BOTTOM)

    model_used = Label(right_frame, text="Model used : 8x8", font=("Arial", 15), bg='white')
    model_used.pack(side=BOTTOM)

    # Bindings
    app.bind('<B1-Motion>', draw)
    app.bind('<ButtonRelease-1>', reset_coords)

    # Main loop
    app.mainloop()

if __name__ == "__main__":
    main()