# Age Guesser Project

A real-time age estimation application that uses your webcam to detect faces and predict their age. The application is built using Python, OpenCV, and a pre-trained Deep Neural Network (Caffe model).

## Features
- **Real-Time Detection**: Accesses your webcam feed to continuously predict ages.
- **Face Detection**: Fast and lightweight face localization using OpenCV's Haar Cascades.
- **Age Prediction**: Employs a Caffe-based Deep Learning model to categorize the detected faces into 8 age ranges: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, and `(60-100)`.
- **Confidence Score**: Displays the confidence percentage of the age prediction.

## Prerequisites
- Python 3.x
- A working webcam

## Installation

1. **Clone or Download the Repository**
   Make sure you are in the project folder (`age_guesser_proj`).

2. **Set up a Virtual Environment (Optional but Recommended)**
   ```sh
   python -m venv venv
   ```
   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

3. **Install Dependencies**
   Install OpenCV and NumPy via the provided `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**
   The application requires the AgeNet Caffe model to function. A script is provided to automatically download the model files to the `models/` directory. Run the following command:
   ```sh
   python download_models.py
   ```
   *This downloads `age_deploy.prototxt` and `age_net.caffemodel`.*

## Usage

1. **Run the Application**
   ```sh
   python app.py
   ```
2. **Interact**
   - The application will open a window showing your webcam feed.
   - Detected faces will be highlighted with a bounding box, displaying the estimated age range and model confidence.
   - **Press `Q`** on your keyboard to quit the application and close the window.

## Troubleshooting

- **Error: Could not find model files**
  Ensure you have run `python download_models.py` successfully and that a `models/` folder exists in your project directory containing the `prototxt` and `caffemodel` files.
- **Webcam not starting**
  Make sure your webcam is connected properly and not currently being used by another application (like Zoom, Teams, or another script).
- **SSL Certificate Errors during model download**
  The script handles SSL verification internally to maximize compatibility on Windows setups, but if errors persist, you might need to check your network constraints or download the models manually.

## Credits
- The age estimation model (`age_net`) used here relies on the `AgeGenderDeepLearning` models by Gil Levi and Tal Hassner.
