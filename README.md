# Brain Tumor MRI Detection
This project is a web application that uses a deep learning model to classify brain tumors from MRI images. Users can upload an MRI scan, and the application will predict whether the image shows a glioma, meningioma, pituitary tumor, or no tumor.


Note: You can upload your screenshot to a site like Imgur and replace the URL above to have it display here.

Features
Image Upload: Simple drag-and-drop or file selection interface.

Deep Learning Model: Utilizes a Convolutional Neural Network (CNN), specifically a pre-trained VGG16 model, fine-tuned for tumor classification.

Classification: Identifies four classes: Glioma, Meningioma, Pituitary, and No Tumor.

Result Display: Shows the uploaded image, the predicted class, and the model's confidence score.

Web Interface: Built with Flask for easy interaction.

**Project Structure**
Brain_Tumor_MRI_Detection/
├── .gitattributes             # Defines how Git handles large files
├── models/
│   └── brain_tumour_detection/
│       └── model.keras        # The trained Keras model (handled by Git LFS)
├── templates/
│   └── index.html             # The HTML front-end
├── uploads/
│   └── (created by the app)   # This folder is ignored by Git
├── main.py                    # The Flask application server
├── requirements.txt           # Python dependencies
└── README.md                  # You are here!

**Technology Stack**
Backend: Python, Flask

Machine Learning: TensorFlow, Keras

Frontend: HTML, Bootstrap CSS

# Setup and Installation
Follow these steps to get the project running on your local machine.

**1. Prerequisites**

Git LFS: The trained model (model.keras) is too large for a standard GitHub repository and is managed using Git Large File Storage (LFS). You must install it first.

Download and install Git LFS.

After installing, run git lfs install once to set it up for your system.

**2. Clone the Repository**
This command will now automatically detect the LFS files and download the model.

git clone https://github.com/Charan282/Brain-Tumor-MRI-Detection.git
cd Brain-Tumor-MRI-Detection

Note: If the model.keras file is only a few KBs in size after cloning, it means Git LFS did not download it correctly. In that case, run git lfs pull inside the repository folder.

**3. Create a Virtual Environment**
It's recommended to use a virtual environment to keep dependencies isolated.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

**4. Install Dependencies**
The required packages are listed in requirements.txt.

pip install -r requirements.txt

How to Run the Application
Once the setup is complete, you can run the Flask application with the following command:

python main.py

Now, open your web browser and navigate to http://127.0.0.1:5000 to use the application.
