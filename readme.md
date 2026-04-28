AI Blood Cancer Detection System
A Machine Learning + Django web application that predicts blood cancer risk using patient data or uploaded medical reports.
Features
. Predict Low / High Risk using ML model
. Displays confidence score (%)
. Uses Random Forest Classifier
. Upload patient report (Image) → OCR extraction
. Manual data entry support
. Hospital-style professional UI
. Confusion matrix during training
. Fast and simple interface

Tech Stack
. Backend: Python, Django
. Machine Learning: Scikit-learn
. OCR: pytesseract, Pillow
. Frontend: HTML, CSS
. Visualization: Matplotlib, Seaborn

Project Structure
blood_app/
│ 
├── manage.py
├── model.pkl 
├── train_model.py 
│ 
├── blood_app/ 
│ ├── settings.py 
│ ├── urls.py 
│ 
├── predictor/ 
│ ├── views.py 
│ ├── urls.py 
│ 
├── predictor/templates/
│ └── index.html 
│ 
├── dataset.csv 
├── confusion_matrix.png


Installation
  1 Clone the repository
  2 Install dependencies
  3 Install Tesseract OCR
  4 Train the model
  5 Run Django server


  Usage
   1 Manual Input
       Enter:

            Age
            Hemoglobin
            WBC
            Platelets

        Click Predict

    2 Upload Report
        Upload medical report image
        System extracts values using OCR
        Automatically predicts risk


Prediction Output
    Low Risk
    High Risk
    Confidence (%)
    Extracted patient values


Model Details
 Algorithm: Random Forest
 Accuracy: 85% – 90%
 Features:
    Age
    Hemoglobin
    WBC
    Platelets


Future Improvements
   PDF report support
   Dashboard analytics
   Medium risk (3-class model)
   User authentication
   Deployment (Render / Heroku)
   Downloadable medical report


Author
   Developed by Pratibha Hadapad