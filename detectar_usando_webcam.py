from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

cap = cv2.VideoCapture(0)

model = YOLO("runs/detect/train/weights/best.pt")

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

while True:
    success, img = cap.read()

    if success:
        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        for result in results:
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  
                        if len(track) > 30: 
                            track.pop(0)

                except:
                    pass

        cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("desligando")