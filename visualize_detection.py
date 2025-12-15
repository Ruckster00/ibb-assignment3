import cv2
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/CelebA-HQ-small.csv')
train_df = df[df['split'] == 'train']

sample_row = train_df.iloc[50]
img_name = f"{sample_row['idx']}.jpg"
img_path = f"data/CelebA-HQ-small/{img_name}"
img = cv2.imread(img_path)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

gt_bbox = [sample_row['x_1'], sample_row['y_1'], 
           sample_row['width'], sample_row['height']]

img_viz = img.copy()
cv2.rectangle(img_viz, 
              (int(gt_bbox[0]), int(gt_bbox[1])), 
              (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])),
              (0, 255, 0), 3)

if len(faces) > 0:
    x, y, w, h = faces[0]
    h_new = int(w * 1.36)
    y_new = y - int((h_new - h) / 2)
    y_new = max(0, y_new)
    cv2.rectangle(img_viz, (x, y_new), (x+w, y_new+h_new), (0, 0, 255), 3)

cv2.putText(img_viz, 'Ground Truth', (int(gt_bbox[0]), int(gt_bbox[1])-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
if len(faces) > 0:
    cv2.putText(img_viz, 'Detected', (x, y_new-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imwrite('results/detection_example.png', img_viz)
print(f"Saved detection visualization for {img_name}")
