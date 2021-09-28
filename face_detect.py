import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while cap.isOpened():
    _, img =  cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(img,1.3, 5)

    for (x, y, w, h) in face:

        if img is not None:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        
    cv2.imshow("window", img)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()