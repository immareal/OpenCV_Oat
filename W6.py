import cv2

# ===== Face Detection =====
face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===== Age Model =====
age_net = cv2.dnn.readNet("age_net.caffemodel","age_deploy.prototxt")

age_list = [
"(0-2)",
"(4-6)",
"(8-12)",
"(15-20)",
"(25-32)",
"(38-43)",
"(48-53)",
"(60-100)"
]

# ===== Gender Model =====
gender_net = cv2.dnn.readNet("gender_net.caffemodel","deploy_gender.prototxt")

gender_list = ["Male","Female"]

# ===== เปิดกล้อง =====
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(
            face_img,1.0,(227,227),
            (78.4263377603,87.7689143744,114.895847746),
            swapRB=False
        )

        # ===== Gender =====
        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward()[0].argmax()]

        # ===== Age =====
        age_net.setInput(blob)
        age = age_list[age_net.forward()[0].argmax()]

        label = f"{gender} {age}"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Age & Gender Detection",frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()