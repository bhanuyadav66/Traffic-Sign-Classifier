import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox, Frame
from PIL import ImageTk, Image
import numpy as np
import os

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

class TrafficSignClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

        self.classes = {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
            9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
            12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
            16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
            19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
            22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
            25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
            28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
            31: 'Wild animals crossing', 32: 'End speed + passing limits',
            33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
            36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
            39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
            42: 'End no passing veh > 3.5 tons'
        }

        self.setup_gui()

    def load_model(self):
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, '..', 'models', 'traffic_classifier_augmented.h5')

        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
            except Exception as e:
                messagebox.showerror("Model Error", f"Error loading model: {e}")
        else:
            messagebox.showerror("Model Not Found", f"Model not found at {model_path}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.geometry('900x800') # Increased window height
        self.root.title('Traffic Sign Recognition System')
        self.root.configure(background='#f0f0f0')
        self.root.resizable(False, False)

        header_frame = Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        Label(header_frame, text="🚦 Traffic Sign Recognition System 🚦", font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50').pack(expand=True)

        self.image_frame = Frame(self.root, bg='#ecf0f1', relief='ridge', bd=2, width=400, height=400)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        self.sign_image = Label(self.image_frame, text="No image selected", font=('Arial', 14), fg='#7f8c8d', bg='#ecf0f1')
        self.sign_image.pack(expand=True)

        button_frame = Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        self.upload_btn = Button(button_frame, text="📁 Upload Image", command=self.upload_image, font=('Arial', 14, 'bold'), bg='#3498db', fg='white', padx=20, pady=10, cursor='hand2')
        self.upload_btn.pack(side='left', padx=10)
        self.classify_btn = Button(button_frame, text="🔍 Classify Sign", command=self.classify_image, font=('Arial', 14, 'bold'), bg='#27ae60', fg='white', padx=20, pady=10, cursor='hand2', state='disabled')
        self.classify_btn.pack(side='left', padx=10)

        result_frame = Frame(self.root, bg='#f8f9fa', relief='ridge', bd=2)
        result_frame.pack(pady=10, padx=50, fill='x')
        Label(result_frame, text="Classification Result:", font=('Arial', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50').pack(pady=(10, 5))
        
        # KEY CHANGE: Added padding and adjusted justification
        self.result_label = Label(result_frame, text="Upload an image and click classify", font=('Arial', 14), bg='#f8f9fa', fg='#34495e', wraplength=700, justify='left')
        self.result_label.pack(pady=(5, 15), padx=10) # Added padding

        self.status_bar = Label(self.root, text="Ready", font=('Arial', 10), fg='green', bg='#f0f0f0', anchor='w')
        self.status_bar.pack(side='bottom', fill='x', padx=10, pady=(0, 5))
        self.current_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.config(state='normal')
            self.result_label.config(text="Image loaded. Click 'Classify Sign'.")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.sign_image.config(image=photo, text="")
        self.sign_image.image = photo

    def classify_image(self):
        if not self.current_image_path: return

        try:
            image = Image.open(self.current_image_path).convert('RGB').resize((30, 30))
            image_array = np.array(image).astype('float32') / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = self.model.predict(image_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]

            sign_name = self.classes.get(predicted_class, f"Unknown")
            result_text = f"🎯 Prediction: {sign_name}\n📊 Confidence: {confidence:.1%}\n\nTop 3 predictions:\n"
            for i, class_idx in enumerate(top_3_indices, 1):
                class_name = self.classes.get(class_idx, f"Class {class_idx}")
                class_confidence = predictions[0][class_idx]
                result_text += f"{i}. {class_name}: {class_confidence:.1%}\n"

            # KEY CHANGE: Ensure text justification is set here as well
            self.result_label.config(text=result_text, justify='left', fg='#27ae60' if confidence > 0.8 else '#f39c12')
            self.root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficSignClassifier()
    app.run()

