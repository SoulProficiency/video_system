from utils.trt_engine_multi import TrtModel
import cv2
infer_engine = TrtModel("./weights/yolo11n_fp16_final.engine",confidence_threshold=0.1,)
cap = cv2.VideoCapture("./data/moving.mp4")
import time

fps = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        t1 = time.time()
        detection = infer_engine.predict(frame,conf=0.1)
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "FPS:%d " % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
        img = infer_engine.draw_detections(frame,detection)
        cv2.imshow("frame", img)
        cv2.waitKey(5)