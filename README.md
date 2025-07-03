# Face Detection using OpenCV Haar Cascade

## 📘 Description

This project demonstrates how to perform **face detection** using the **Haar Cascade Classifier** from the OpenCV library in Python. It is a classic and efficient technique for identifying faces in images using machine learning-based classifiers that have been trained to detect faces.

The script loads a static image, processes it to grayscale, detects facial features, and displays both the grayscale version and the original image annotated with rectangles around detected faces.

---

## 🧾 Features

- Uses **OpenCV**'s pre-trained `haarcascade_frontalface_default.xml` classifier.
- Detects faces in static images.
- Visual feedback through GUI windows using `cv2.imshow()`.
- Highlights detected faces with bounding boxes.

---

## 🖥️ Requirements

Ensure the following are installed:

- Python 3.x
- OpenCV (`opencv-python` package)

You can install OpenCV with pip:

```bash
pip install opencv-python
```

---

## 🗂️ File Structure

- `haarcascade_frontalface_default.xml` – The XML file containing the pre-trained Haar cascade face detection data.
- `img.webp` – Sample image to test face detection.
- `face_detection.py` – Python script performing detection.
- `README.md` – Project documentation (this file).

---

## ▶️ How It Works

### 1. Load the Haar Cascade Classifier

```python
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

This loads the face detection model.

### 2. Read and Convert the Image to Grayscale

```python
image = cv2.imread('img.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Haar cascade works best with grayscale images for faster processing.

### 3. Detect Faces in the Grayscale Image

```python
faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
```

- `scaleFactor=1.1`: Specifies how much the image size is reduced at each scale.
- `minNeighbors=4`: Defines how many neighbors each candidate rectangle should have to retain it.

### 4. Draw Rectangles Around Detected Faces

```python
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)
```

### 5. Display the Results

```python
cv2.imshow("Gray", gray)
cv2.imshow("Faces", image)
cv2.waitKey()
```

Press any key while the window is focused to close it.

---

## 🚫 Limitations

- May not work effectively in poor lighting or low-resolution images.
- Only suitable for front-facing, upright faces due to the Haar training dataset.
- Performance can degrade with occlusions (glasses, hats, etc.).

---

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Haar Cascade Tutorial](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html)
- [Pre-trained Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)

---

## 📌 Author

This face detection project is a simple and effective demonstration of traditional computer vision techniques using OpenCV.