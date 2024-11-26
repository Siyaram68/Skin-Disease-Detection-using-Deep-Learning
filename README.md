# 🩺 Skin Disease Detection using Deep Learning

**Overview**

This project leverages the power of deep learning to detect and classify skin diseases from images. By analyzing patterns and features in medical images, the model can provide an initial assessment that aids healthcare professionals in diagnosing conditions more efficiently.

This system is designed to work as a decision-support tool and is not a substitute for professional medical advice.

**Features**

- 🖼️ Image Classification: Detect multiple skin disease types from input images.
- 🧠 Deep Learning Model: Built using convolutional neural networks (CNNs) for high accuracy.
- 📊 Performance Metrics: Includes accuracy, precision, recall, and F1-score.
- 🌐 User Interface: Streamlit is  simple GUI interface to upload images for detection.

**Technologies Used**

- Deep Learning Framework: TensorFlow / PyTorch
- Programming Language: Python
- Libraries: NumPy, Pandas, Matplotlib, OpenCV
- Dataset: Kaggle(HAM 10000 Skin Lesion Dataset)
- Tools: Jupyter Notebook / Google Colab

**How it Works**

Dataset Preprocessing:

- Images are resized, normalized, and augmented for training.
- Data is split into training, validation, and test sets.
- Model Architecture:
- A CNN model with layers optimized for feature extraction.
- Utilizes techniques like dropout, batch normalization, and early stopping to prevent overfitting.
  Training:
- The model is trained on labeled images using categorical cross-entropy loss.
- Hyperparameters like learning rate, batch size, and number of epochs are fine-tuned for optimal performance.

**Prediction:**

Input images are processed, and the model outputs the predicted disease category with confidence scores.

**How to Run**

### **Setup Environment**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/skin-disease-detection.git
   cd skin-disease-detection

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Run the model:**
  3.1  **Train the Model:**

   ```bash
               python train.py
  
3.2 **Test the Model:**
    
    ```bash
              python test.py
 
 3.3 **Run Predictions:**
    
     ```bash
                python predict.py --image path/to/image.jpg

**Results**

- Accuracy: 89.52% on the test set.
- Precision: 84.78%
- Recall: 81.95%
- F1-Score: 81.14%

**Sample predictions with visualizations:**

Healthy Skin: ✔️
Psoriasis: ❌
Melanoma: ❌

**Future Enhancements**

- Incorporate more skin disease categories for broader applicability.
- Improve the user interface for non-technical users.
- Integrate with mobile apps for on-the-go diagnosis.
- Train the model on a larger, more diverse dataset to reduce bias.

  
**Disclaimer**

This tool is intended for educational purposes and should not be used for medical decision-making without consulting a healthcare professional.










