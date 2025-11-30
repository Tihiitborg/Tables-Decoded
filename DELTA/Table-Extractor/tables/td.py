import os
# import glob
# import tqdm
import cv2
import numpy as np
from ultralytics import YOLO
# from PIL import Image, ImageDraw
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

class TableDetector:
    def __init__(
            self, 
            model= os.path.join(CURRENT_DIR, "model", "yolo-td.pt"), 
            conf_thresh=0.75
            ) -> None:
        self.model = YOLO(model)
        self.model.overrides["iou"]=0.1
        self.conf_thresh= conf_thresh

    
    def predict(self, image, thresh = 0.5):
        if isinstance(image, str):
            image = cv2.imread(image, 0)
            orig_image = image.copy()
            # BGR to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        elif isinstance(image, np.ndarray):
            orig_image = image.copy()
            # BGR to RGB
            image = orig_image.astype(np.float32)
        # Resize the image
        height, width, _ = image.shape

        dets = []
        results = self.model(image, save=False, show_labels=True, show_conf=True, show_boxes=True, conf = self.conf_thresh)
        for entry in results:
            bboxes = entry.boxes.xyxy.cpu().numpy()
            # classes = entry.boxes.cls.cpu().numpy()
            conf = entry.boxes.conf.cpu().numpy()
            for i in range(len(bboxes)):
                box = bboxes[i]
                if conf[i] > thresh:
                    dets.append([box[0], box[1], box[2], box[3]])
        return dets

if __name__=="__main__":
    td = TableDetector()
    image = "samples/sample3.jpeg"
    dets = td.predict(image)
    print(dets)
