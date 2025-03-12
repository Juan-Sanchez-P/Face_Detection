from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import time

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load cascade classifiers once at startup
cascades = {
    "smile": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml"),
    "face": cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ),
    "face_alt": cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    ),
    "face_alt2": cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    ),
    "profile": cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    ),
    "eye": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml"),
}

# Verify all classifiers loaded properly
for name, classifier in cascades.items():
    if classifier.empty():
        print(f"Warning: {name} classifier failed to load")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template(
        "index.html", uploaded_image=None, processed_image=None, face_count=None, smile_count=None
    )

@app.route("/detect", methods=["POST"])
def detect():
    # Check if an image file was uploaded
    if "image" not in request.files:
        flash("No image part in the request")
        return redirect(url_for("index"))
        
    file = request.files["image"]
    
    if file.filename == "":
        flash("No image selected")
        return redirect(url_for("index"))
        
    if not allowed_file(file.filename):
        flash("File type not allowed. Please use jpg, jpeg, png, or gif")
        return redirect(url_for("index"))
    
    # Secure the filename to prevent path traversal attacks
    secure_fname = secure_filename(file.filename)
    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_fname}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        file.save(filepath)
        
        # Process image and get results
        processed_filepath, face_count, smile_count = detect_faces(filepath, filename)
        
        # Prepare image URLs for template
        uploaded_image_url = url_for("static", filename=f"uploads/{filename}")
        processed_image_url = (
            url_for("static", filename=f"processed/{filename}")
            if face_count > 0
            else None
        )
        
        return render_template(
            "index.html",
            uploaded_image=uploaded_image_url,
            processed_image=processed_image_url,
            face_count=face_count,
            smile_count=smile_count,
        )
    except Exception as e:
        flash(f"Error processing image: {str(e)}")
        return redirect(url_for("index"))

def merge_overlapping_faces(faces, overlap_threshold=0.3):
    """Merge face rectangles that overlap significantly"""
    if len(faces) == 0:
        return np.array([], dtype=np.int32).reshape(0, 4)
        
    # Convert to list of tuples for processing
    faces_list = [(x, y, w, h) for x, y, w, h in faces]
    result = []

    while faces_list:
        current = faces_list.pop(0)
        x1, y1, w1, h1 = current

        i = 0
        while i < len(faces_list):
            x2, y2, w2, h2 = faces_list[i]

            # Calculate overlap area
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            min_area = min(w1 * h1, w2 * h2)

            # Merge if overlap is significant
            if overlap_area > overlap_threshold * min_area:
                # Create a bounding box that encompasses both faces
                x1, y1 = min(x1, x2), min(y1, y2)
                w1 = max(x1 + w1, x2 + w2) - x1
                h1 = max(y1 + h1, y2 + h2) - y1
                faces_list.pop(i)
            else:
                i += 1

        result.append((x1, y1, w1, h1))

    return np.array(result, dtype=np.int32)

def detect_faces(image_path, filename):
    """Detect faces and smiles in an image using multiple cascades."""
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, 0  # No image found, return zero faces & smiles

    # Resize large images for performance
    max_dimension = 1200
    height, width = img.shape[:2]
    scale_factor = 1.0
    
    if max(height, width) > max_dimension:
        scale_factor = max_dimension / max(height, width)
        img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for better detection
    gray = cv2.equalizeHist(gray)

    # Detect faces using multiple cascades
    faces = cascades["face"].detectMultiScale(gray, 1.1, 8, minSize=(30, 30))

    if len(faces) == 0:
        faces = cascades["face_alt"].detectMultiScale(gray, 1.1, 8, minSize=(30, 30))
        
    if len(faces) == 0:
        faces = cascades["face_alt2"].detectMultiScale(gray, 1.1, 6, minSize=(30, 30))
    
    # If we found multiple faces, merge overlapping detections
    if len(faces) > 1:
        faces = merge_overlapping_faces(faces)

    # Track smile count
    smile_count = 0

    # Process detected faces
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Create region of interest focused on lower half of face
        # This is where smiles typically occur
        roi_y = y + int(h * 0.4)  # Start from the lower 60% of the face
        roi_h = int(h * 0.6)      # Only check the lower 60% of the face
        
        # Make sure the ROI stays within the image bounds
        roi_y = max(0, roi_y)
        roi_h = min(roi_h, img.shape[0] - roi_y)
        
        if roi_h <= 0 or w <= 0:  # Skip if ROI is invalid
            continue
            
        roi_gray = gray[roi_y:roi_y + roi_h, x:x + w]
        
        # Use less strict parameters for smile detection
        smiles = cascades["smile"].detectMultiScale(
            roi_gray, 
            scaleFactor=1.3,     # Increased from 1.1 to allow more scale variation
            minNeighbors=15,     # Reduced from 30 to allow more detections
            minSize=(int(w/6), int(h/10))  # Adaptive sizing based on face dimensions
        )

        for sx, sy, sw, sh in smiles:
            # Adjust coordinates to match the ROI offset
            cv2.rectangle(
                img, 
                (x + sx, roi_y + sy), 
                (x + sx + sw, roi_y + sy + sh), 
                (255, 0, 0), 
                2
            )
            smile_count += 1

    # Save processed image if faces were detected
    processed_filepath = None
    if len(faces) > 0:
        processed_filepath = os.path.join(app.config["PROCESSED_FOLDER"], filename)
        cv2.imwrite(processed_filepath, img)

    return processed_filepath, len(faces), smile_count

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
