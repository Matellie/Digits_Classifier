from models import *

from tkinter import *
from PIL import ImageGrab, Image, ImageTk

def main():

    def draw(event):
        x, y = event.x, event.y
        if canvas.old_coords:
            x1, y1 = canvas.old_coords
            canvas.create_line(x, y, x1, y1, width=50, fill='white')
        canvas.old_coords = x, y

        image_8x8px, image_display = get_images()
        display_image(image_display)

        #predict(image_8x8px)
        #display_prediction()

    def reset_coords(event):
        canvas.old_coords = None

    def get_images():
        x1 = app.winfo_rootx() + canvas.winfo_x() + left_frame.winfo_x()
        y1 = app.winfo_rooty() + canvas.winfo_y() + left_frame.winfo_y()
        x2 = x1 + canvas.winfo_width()
        y2 = y1 + canvas.winfo_height()
        image = ImageGrab.grab().crop((x1+5, y1+5, x2-5, y2-5))

        image_8x8px = image.resize((8, 8))
        image_display = image_8x8px.resize((150, 150), Image.NEAREST)

        return image_8x8px, image_display
    
    def display_image(image):
        image = ImageTk.PhotoImage(image)
        img_panel.configure(image=image)
        img_panel.image = image

    def save_image(image):
        image.save('image.png')

    def predict(image):
        pass
    
    def display_prediction(pred):
        pass

    app = Tk()
    app.geometry("800x500")
    app.title("Digit Detector")
    app.configure(bg='white')

    app_title = Label(app, text="Digit Detector\n", font=("Arial", 20), bg='white')
    app_title.pack()

    left_frame = Frame(app, bg='white')
    left_frame.pack(side=LEFT)

    draw_area_title = Label(left_frame, text="⬇ Draw a digit below ⬇", font=("Arial", 15), bg='white')
    draw_area_title.pack()

    canvas = Canvas(left_frame, width=400, height=400)
    canvas.configure(bg='black')
    canvas.pack()
    canvas.old_coords = None

    right_frame = Frame(app, bg='white')
    right_frame.pack()

    img_display_title = Label(right_frame, text="Image used by the model", font=("Arial", 15), bg='white')
    img_display_title.pack()

    img = ImageTk.PhotoImage(Image.new('RGB', (150, 150), color='black'))
    img_panel = Label(right_frame, image=img)
    img_panel.image = img
    img_panel.pack()

    pred_area_title = Label(right_frame, text="\n⬇ Prediction ⬇", font=("Arial", 15), bg='white')
    pred_area_title.pack()

    pred_area = Label(right_frame, text="None", font=("Arial", 20), bg='white')
    pred_area.pack()

    model = SimpleNeuralNet(input_size=64, hidden_size=32, nb_classes=10)
    model = load_model(model=model, model_path='model_ckpt', model_name='model_MNISTDigits_SimpleNeuralNet_97.pt')

    

    app.bind('<B1-Motion>', draw)
    app.bind('<ButtonRelease-1>', reset_coords)

    app.mainloop()

if __name__ == "__main__":
    main()