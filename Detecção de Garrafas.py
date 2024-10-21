from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0) 

model = YOLO("runs/detect/train/weights/best.pt")  

seguir = True  

while True:
    success, img = cap.read()  

    if success:
        
        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

       
        for result in results:
            img = result.plot()

        cv2.imshow("Tela", img)  

    k = cv2.waitKey(1) 
    if k == ord('q'):  
        break

cap.release()  
cv2.destroyAllWindows() 
print("desligando")  
