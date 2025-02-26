from flask import Flask, render_template, request, redirect, url_for
import os
import cv2

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load cascade classifiers
cascades = {
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


@app.route("/")
def index():
    return render_template(
        "index.html", uploaded_image=None, processed_image=None, face_count=None
    )


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files or request.files["image"].filename == "":
        return redirect(request.url)

    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Process image and get results
    processed_filepath, face_count = detect_faces(filepath, file.filename)

    # Prepare image URLs for template
    uploaded_image_url = url_for("static", filename=f"uploads/{file.filename}")
    processed_image_url = (
        url_for("static", filename=f"processed/{file.filename}")
        if face_count > 0
        else None
    )

    return render_template(
        "index.html",
        uploaded_image=uploaded_image_url,
        processed_image=processed_image_url,
        face_count=face_count,
    )


def merge_overlapping_faces(faces, overlap_threshold=0.3):
    """Merge face rectangles that overlap significantly"""
    if not faces:
        return []

    faces_list = list(faces)
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
                x1, y1 = min(x1, x2), min(y1, y2)
                w1, h1 = max(x1 + w1, x2 + w2) - x1, max(y1 + h1, y2 + h2) - y1
                faces_list.pop(i)
            else:
                i += 1

        result.append((x1, y1, w1, h1))

    return result


def detect_faces(image_path, filename):
    """Multi-cascade face detection for improved accuracy"""
    img = cv2.imread(image_path)
    if img is None:
        return None, 0

    # Prepare image for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    faces = []

    # Try frontal face detection with different cascades
    faces_default = cascades["face"].detectMultiScale(gray, 1.1, 8, minSize=(30, 30))
    if len(faces_default) == 0:
        faces.extend(
            cascades["face_alt"].detectMultiScale(gray, 1.1, 8, minSize=(30, 30))
        )
    else:
        faces.extend(faces_default)

    # Try with alt2 cascade
    faces_alt2 = cascades["face_alt2"].detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    faces.extend(
        [
            face
            for face in faces_alt2
            if not any(
                abs(face[0] - f[0]) < 20 and abs(face[1] - f[1]) < 20 for f in faces
            )
        ]
    )

    # Try profile detection in both orientations
    profile_faces = cascades["profile"].detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

    # Detect profiles in flipped image
    flipped = cv2.flip(gray, 1)
    flipped_profiles = cascades["profile"].detectMultiScale(
        flipped, 1.1, 5, minSize=(30, 30)
    )

    # Convert flipped coordinates back
    img_width = img.shape[1]
    for x, y, w, h in flipped_profiles:
        faces.append((img_width - x - w, y, w, h))

    # Add profile faces (excluding duplicates)
    faces.extend(
        [
            face
            for face in profile_faces
            if not any(
                abs(face[0] - f[0]) < 20 and abs(face[1] - f[1]) < 20 for f in faces
            )
        ]
    )

    # Merge overlapping detections
    faces = merge_overlapping_faces(faces, 0.4)
    face_count = len(faces)

    if face_count > 0:
        for x, y, w, h in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Detect and draw eyes
            # roi_gray = gray[y : y + h, x : x + w]
            # eyes = cascades["eye"].detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
            # for ex, ey, ew, eh in eyes:
            #     cv2.rectangle(
            #         img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2
            #     )
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        processed_filepath = os.path.join(app.config["PROCESSED_FOLDER"], filename)
        cv2.imwrite(processed_filepath, img)

    return os.path.join(
        app.config["PROCESSED_FOLDER"], filename
    ) if face_count > 0 else None, face_count


if __name__ == "__main__":
    app.run(debug=True)
