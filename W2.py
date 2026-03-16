import cv2

video = cv2.VideoCapture("video.avi")

fps = video.get(cv2.CAP_PROP_FPS)  # จำนวนเฟรมต่อวินาที
frame_count = 0
img_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # บันทึกทุก ๆ 1 วินาที
    if frame_count % int(fps) == 0:
        cv2.imwrite(f"output{img_count}.jpg", frame)
        img_count += 1

    frame_count += 1

video.release()
print("เสร็จแล้ว")

