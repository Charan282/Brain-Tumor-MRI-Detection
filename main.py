import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load the Trained Model ---
# Ensure this path is correct relative to where you run the flask app.
# Based on your screenshot, this path seems correct if you run `python main.py` from the project root.
try:
    model = load_model('models/brain_tumour_detection/model.keras')
except Exception as e:
    print(f"***** Error loading model: {e} *****")
    # Handle the error appropriately, maybe exit or use a dummy model
    model = None

# --- Configuration ---
# Define the uploads folder and ensure it exists.
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- (FIXED) Class Label Correction ---
# Your model was trained with class labels in alphabetical order.
# This list MUST match the order from your training notebook, which was:
# class_labels = sorted(os.listdir(train_dir))
#
# The correct order is:
# 0: glioma
# 1: meningioma
# 2: notumor
# 3: pituitary
# The list below reflects this correct order.
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


# --- Image Preprocessing Function ---
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Loads and preprocesses the uploaded image to match the model's expected input format.
    This matches the preprocessing from your 'detect_and_display' function in the notebook.
    """
    try:
        # Load the image, resizing to 128x128 with 3 color channels (RGB)
        img = load_img(image_path, target_size=target_size, color_mode='rgb')

        # Convert the image to a numpy array
        img_array = img_to_array(img)

        # Normalize the image data to be between 0 and 1
        img_array = img_array / 255.0

        # Add a batch dimension because the model expects it (shape: 1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        print(f"***** Error preprocessing image: {e} *****")
        return None

# --- Prediction Function ---
def predict_tumor(image_path):
    """
    Takes an image path, preprocesses it, and returns the model's prediction
    and the confidence score.
    """
    if model is None:
        return "Model is not loaded. Please check server logs.", 0.0

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None:
        return "Error during image processing. Check file format.", 0.0

    # Get model predictions (it returns a list of probabilities)
    predictions = model.predict(preprocessed_img)

    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    # Get the highest probability value
    confidence = np.max(predictions[0])

    # Get the corresponding class label from our corrected list
    predicted_class_label = class_labels[predicted_class_index]

    # Format the result string for display
    if predicted_class_label == 'notumor':
        result_text = 'No Tumor Detected'
    else:
        # Capitalize the first letter for better display
        result_text = f'Tumor Detected: {predicted_class_label.capitalize()}'

    return result_text, confidence


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was sent with the request
        if 'file' not in request.files:
            # This is a good practice to handle cases where the form is submitted empty
            return render_template('index.html', error='No file part in the request.')
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error='No file selected for uploading.')

        if file:
            # Use secure_filename to prevent malicious file names
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get prediction and confidence
            result, confidence = predict_tumor(file_path)

            # Render the page with the prediction results
            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}",
                file_path=f'/uploads/{filename}'
            )

    # This is for the initial page load (GET request)
    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image file so it can be displayed in the HTML."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Run the App ---
if __name__ == '__main__':
    # Using host='0.0.0.0' makes the app accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)
