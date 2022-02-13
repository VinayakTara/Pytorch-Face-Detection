import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255),
                              thickness=2)

                cv2.circle(frame, (int(ld[0][0]),int(ld[0][1])), 5, (0,0,255), 1)
                cv2.circle(frame, (int(ld[1][0]),int(ld[1][1])), 5, (0, 0, 255), 1)
                cv2.circle(frame, (int(ld[2][0]),int(ld[2][1])), 5, (0, 0, 255), 1)
                cv2.circle(frame, (int(ld[3][0]),int(ld[3][1])), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(ld[4][0]),int(ld[4][1])), 5, (0, 0, 255), -1)
                print("\n")
                print("left eye coordinate:",int(ld[0][0]),int(ld[0][1]))
                print("right eye coordinate:",int(ld[1][0]),int(ld[1][1]))
                print("nose coordinate:",int(ld[2][0]),int(ld[2][1]))
                print("left mouth coordinate:",int(ld[3][0]),int(ld[3][1]))
                print("right mouth coordinate:",int(ld[4][0]),int(ld[4][1]))
        except Exception as e:
            print(e)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self._draw(frame, boxes, probs, landmarks)
            except:
                pass
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()