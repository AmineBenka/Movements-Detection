import cv2

# Charger modèle pré-entraîné (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Camera non accessible")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20)
    )

    # Dessiner rectangles
    for (x, y, w, h) in faces:
        print(f"Face détectée à: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()