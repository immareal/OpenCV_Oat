import cv2

# face detection
face = cv2.CascadeClassifier(cv2.data.haarcascades +
                             "haarcascade_frontalface_default.xml")

# gender model
gender_net = cv2.dnn.readNet(
    "gender_net.caffemodel",
    "deploy_gender.prototxt"
)

gender_list = ["Male","Female"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(
            face_img,1.0,(227,227),
            (78.4263377603,87.7689143744,114.895847746),
            swapRB=False
        )

        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward()[0].argmax()]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,gender,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Gender Detection",frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()