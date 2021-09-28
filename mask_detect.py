#pretrained model  : densenet
# 1 : face mask
# 0 : no mask



import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import datetime
from tensorflow.keras.models import load_model

mymodel=load_model('face_model_CNN.h5')

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        # test_image=image.load_img('temp.jpg',target_size=(150,150,3))

        # print("image shape : ", face_img.shape)
        img = img / 255.0
        test_image = cv2.resize(face_img, (150, 150), interpolation=cv2.INTER_CUBIC)
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)


        prediction=mymodel.predict(test_image)[0]
        pred = np.argmax(prediction)

        # print(f"prediction {pred}")
        if pred==0:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK!',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()