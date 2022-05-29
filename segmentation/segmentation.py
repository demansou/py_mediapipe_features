import cv2
import mediapipe as mp
import numpy as np

class Segmentation:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.segmenter = mp.solutions.selfie_segmentation
    
    def segment(self, segment_bg, logging=False):
        cap = cv2.VideoCapture(0)
        with self.segmenter.SelfieSegmentation(model_selection=1) as seg:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if logging:
                        print("Ignoring empty camera frame.")
                    continue
                image, results = self.__mediapipe_detection(frame, seg)
                if logging:
                    print(results)
                image = self.__segment(image, results, segment_bg)
                cv2.imshow("OpenCV Feed", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    
    def __mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def __segment(self, image, results, segment_bg):
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        if segment_bg is None:
            segment_bg = np.zeros(image.shape,  dtype=np.uint8)
            segment_bg[:] = (192, 192, 192)
        else:
            segment_bg = cv2.imread(segment_bg)
        image = np.where(condition, image, segment_bg)
        return image
