import cv2

# โหลดโมเดล
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

nose = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
mouth = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

# ตรวจสอบว่าโหลดได้ไหม
if nose.empty():
    print("โหลด nose cascade ไม่ได้")
if mouth.empty():
    print("โหลด mouth cascade ไม่ได้")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    # นับจำนวนหน้า
    face_count = len(faces)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,"Face",(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # eye
        eyes = eye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:

            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            brightness = eye_img.mean()

            if brightness < 70:
                label = "Glasses"
            else:
                label = "No Glasses"

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(roi_color,label,(ex,ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        # nose
        if not nose.empty():
            noses = nose.detectMultiScale(roi_gray)
            for (nx,ny,nw,nh) in noses:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
                cv2.putText(roi_color,"Nose",(nx,ny-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        # mouth
        if not mouth.empty():
            mouths = mouth.detectMultiScale(roi_gray)
            for (mx,my,mw,mh) in mouths:
                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(255,255,0),2)
                cv2.putText(roi_color,"Mouth",(mx,my-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

    # แสดงจำนวนหน้ามุมขวาล่าง
    text = f"Faces: {face_count}"
    h, w = frame.shape[:2]

    cv2.putText(frame,text,(w-180,h-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,255,255),2)

    cv2.imshow("Face Detection",frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()