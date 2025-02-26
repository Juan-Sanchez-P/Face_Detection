from flask import Flask, render_template, request, redirect, url_for
import os
import cv2

app = Flask(__name__)

# Set the upload and processed folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None, processed_image=None, face_count=None)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Perform face detection
        processed_filepath, face_count = detect_faces(filepath, file.filename)

        # Convert paths to be accessible in HTML
        uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')
        processed_image_url = url_for('static', filename=f'processed/{file.filename}') if face_count > 0 else None

        return render_template(
            'index.html',
            uploaded_image=uploaded_image_url,
            processed_image=processed_image_url,
            face_count=face_count
        )

def detect_faces(image_path, filename):
    """Detects faces in the uploaded image and saves a processed version with rectangles drawn around faces."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_count = len(faces)  # Count detected faces

    if face_count > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Draw rectangle around faces

        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cv2.imwrite(processed_filepath, img)  # Save the processed image
        return processed_filepath, face_count
    else:
        return None, 0  # No faces detected

if __name__ == '__main__':
    app.run(debug=True)
