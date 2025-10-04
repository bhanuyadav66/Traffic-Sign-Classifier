# 🚦 Traffic Sign Recognition System

This project is a deep learning application that identifies German traffic signs from images. It uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to achieve high accuracy and features a user-friendly graphical interface created with Tkinter.


## ✨ Features

-   **High Accuracy**: Employs a CNN model trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, achieving over 99% validation accuracy.
-   **Data Augmentation**: The model is trained with augmented data (rotations, zooms, shifts) to make it more robust and to prevent overfitting, improving its performance on real-world images.
-   **Interactive GUI**: A simple and intuitive graphical user interface built with Tkinter allows users to upload an image and receive an instant classification.
-   **Top Predictions**: Displays the top 3 most likely predictions along with their confidence scores, giving a more nuanced view of the model's decision process.

## 🛠️ Technologies Used

-   **Backend & Model**: Python, TensorFlow, Keras
-   **GUI**: Tkinter
-   **Data Handling**: NumPy, Pandas, Pillow (PIL)
-   **Plotting**: Matplotlib

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9+
-   `pip` package manager
-   `venv` module for virtual environments

### Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/bhanuyadav66/traffic-sign-recognition.git
    cd traffic-sign-recognition
    ```

2.  **Create and activate a virtual environment:**
    ```
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    -   Download the German Traffic Sign Recognition Benchmark (GTSRB) dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
    -   Extract the contents and place the `Train` and `Test` folders inside the `data/` directory. Your structure should look like `data/Train/` and `data/Test/`.

### How to Run

1.  **Train the Model:**
    -   Run the training script from the root directory. This will train the model with data augmentation and save it to `models/traffic_classifier_augmented.h5`.
    ```
    python -m scripts.train_model
    ```

2.  **Launch the GUI Application:**
    -   Once the model is trained, run the GUI script.
    ```
    python -m scripts.gui_app
    ```
    -   Click the "Upload Image" button, select an image of a traffic sign, and click "Classify Sign" to see the prediction.

## 📁 Project Structure

traffic-sign-recognition/
├── data/ # (To be created by user) For dataset files
├── models/ # Trained model will be saved here
├── outputs/ # Training history plots will be saved here
├── scripts/
│ ├── train_model.py # Script to train the CNN model
│ └── gui_app.py # Script for the Tkinter GUI application
├── .gitignore # Specifies files for Git to ignore
├── README.md # Project documentation
└── requirements.txt # Project dependencies


## 📈 Results

The model was trained for 30 epochs with data augmentation, achieving the following performance on the validation set:
-   **Validation Accuracy**: ~99.6%
-   **Validation Loss**: ~0.014

The training process demonstrated the model's ability to learn distinguishing features effectively, although testing revealed challenges in differentiating visually similar signs—a common problem in image classification that highlights the importance of robust data augmentation.
Few classification outputs:
<img width="1119" height="1016" alt="image" src="https://github.com/user-attachments/assets/8e7a5a4c-1167-47dd-8023-68424adda4c3" />


## Future Improvements

-   **Implement Advanced Augmentation**: Use techniques like random erasing or CutMix to force the model to focus on more subtle features inside the signs.
-   **Optimize Model Architecture**: Experiment with different CNN architectures (e.g., ResNet, MobileNet) for potentially better performance or faster inference.
-   **Real-time Video Classification**: Extend the application to use a webcam feed for real-time traffic sign detection and classification.

