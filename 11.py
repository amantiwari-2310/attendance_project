import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
import datetime

# Function to create a dataset
def create_dataset(name, student_id):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create directory if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # Create timestamp file
    timestamp_file = open(f"dataset/{name}_{student_id}_timestamps.txt", 'w')

    # Counter for images captured
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Save the captured face with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"dataset/{name}_{student_id}_{count}_{timestamp}.jpg", gray[y:y+h, x:x+w])
            timestamp_file.write(f"{count}_{timestamp}\n")
            count += 1

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or count >= 50:
            break

    timestamp_file.close()
    cap.release()
    cv2.destroyAllWindows()

    # Return the number of images captured
    return count

# Function to train the model
def train_model():
    # Load images and labels
    faces = []
    labels = []

    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)

                # Parse filename to extract label (student ID)
                try:
                    parts = file.split("_")
                    if len(parts) < 3:
                        continue  # Skip files that don't match expected naming convention

                    label = int(parts[1])  # Extract student ID from filename
                    img = cv2.imread(path, 0)
                    faces.append(img)
                    labels.append(label)
                except ValueError:
                    continue  # Skip files where student ID is not a valid integer

    # Create LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    recognizer.train(faces, np.array(labels))

    return recognizer

# Function to recognize face using a trained model
def recognize_faces_with_model(recognizer):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load timestamps for matching
    timestamp_dict = {}
    access_log_set = set()

    # Create access log file
    access_log_file = open("access_log.txt", 'a')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Recognize the face
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:  # You may need to adjust this threshold based on your dataset
                # Check if student ID has already been logged
                if id_ not in access_log_set:
                    # Access granted
                    cv2.putText(frame, f"Access granted. Student ID: {id_}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Log timestamp when access is granted
                    access_log_file.write(f"Student ID: {id_}, Timestamp: {datetime.datetime.now()}\n")
                    access_log_set.add(id_)  # Add student ID to the set to prevent duplicate logging
            else:
                # Access denied
                cv2.putText(frame, "Access denied", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    access_log_file.close()
    cap.release()
    cv2.destroyAllWindows()

# Main GUI class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.configure(background='white')  # Set background color

        self.label_name = tk.Label(root, text="Enter student name:", bg='white', fg='blue')
        self.label_name.pack()
        self.entry_name = tk.Entry(root)
        self.entry_name.pack()

        self.label_id = tk.Label(root, text="Enter student ID:", bg='white', fg='blue')
        self.label_id.pack()
        self.entry_id = tk.Entry(root)
        self.entry_id.pack()

        self.capture_button = tk.Button(root, text="Capture Images", command=self.capture_images, bg='blue', fg='white')
        self.capture_button.pack()

        self.recognize_button = tk.Button(root, text="Recognize Face", command=self.recognize_faces, bg='blue', fg='white')
        self.recognize_button.pack()

        # Load the recognizer model
        self.recognizer = train_model()

    def capture_images(self):
        name = self.entry_name.get()
        student_id = self.entry_id.get()

        if name == '' or student_id == '':
            messagebox.showwarning("Warning", "Please enter both name and ID.")
            return

        count = create_dataset(name, student_id)
        messagebox.showinfo("Information", f"{count} images captured.")

        # Save user input to a file
        with open('user_input.txt', 'w') as f:
            f.write(f"Name: {name}\nID: {student_id}\n")

    def recognize_faces(self):
        recognize_faces_with_model(self.recognizer)

# Main function
def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
