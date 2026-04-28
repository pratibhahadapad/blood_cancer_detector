import joblib
import numpy as np
import os
import re
from django.shortcuts import render
from PIL import Image
import pytesseract


# TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)


# OCR FUNCTION
def extract_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


# VALUE EXTRACTION
def extract_values(text):
    hb = re.search(r'Hemoglobin[:\s]*([\d.]+)', text)
    wbc = re.search(r'WBC[:\s]*([\d]+)', text)
    platelets = re.search(r'Platelet[:\s]*([\d]+)', text)

    return (
        float(hb.group(1)) if hb else 13,
        int(wbc.group(1)) if wbc else 7000,
        int(platelets.group(1)) if platelets else 250000
    )


# MAIN VIEW
def home(request):
    result = None
    confidence = None
    extracted = None

    if request.method == "POST":

      # CASE 1: MANUAL INPUT        
        if request.POST.get('age'):
            age = float(request.POST['age'])
            hb = float(request.POST['hemoglobin'])
            wbc = float(request.POST['wbc'])
            platelets = float(request.POST['platelets'])

       
        # CASE 2: IMAGE UPLOAD (OCR)
        elif request.FILES.get('report'):
            file = request.FILES['report']
            file_path = os.path.join(BASE_DIR, "temp.png")

            with open(file_path, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            text = extract_text(file_path)

            hb, wbc, platelets = extract_values(text)
            age = 30  # default

            extracted = {
                "hb": hb,
                "wbc": wbc,
                "platelets": platelets
            }

        else:
            return render(request, "index.html")

    
        # ML PREDICTION
        data = np.array([[age, hb, wbc, platelets]])

        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        confidence = max(proba) * 100

    
        if hb > 13 and wbc < 8000 and platelets > 90000:
            prediction = "Low"
            confidence = max(confidence, 85)  # boost confidence slightly

        result = prediction

    return render(request, "index.html", {
        "result": result,
        "confidence": round(confidence, 2) if confidence else None,
        "extracted": extracted
    })