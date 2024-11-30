import cv2

camera = cv2.VideoCapture('video.mp4')
classificador = cv2.CascadeClassifier(r'haarcascade_fullbody.xml')

while True:
    check,img = camera.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    objetos = classificador.detectMultiScale(imgGray,minSize=(30,30),scaleFactor=1.5)
    # print(objetos)
    for x,y,l,a in objetos:
        cv2.rectangle(img,(x,y),(x+l,y+a),(255,0,0),2)
    cv2.imshow('imagem',img)
    cv2.waitKey(1)