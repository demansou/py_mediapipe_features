import cv2
import mediapipe as mp

class MediaPipeDetection:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, logging=False):
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.__mediapipe_detection(frame, holistic)
                if logging:
                    print(results)
                self.__draw_styled_landmarks(image, results)
                cv2.imshow('OpenCV Feed', image)
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
    
    def __draw_styled_landmarks(self, image, results):
        drawer = self.mp_drawing
        connections = self.mp_holistic
        drawer.draw_landmarks(image, results.pose_landmarks, connections.POSE_CONNECTIONS,
            drawer.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            drawer.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        drawer.draw_landmarks(image, results.left_hand_landmarks, connections.HAND_CONNECTIONS,
            drawer.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            drawer.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        drawer.draw_landmarks(image, results.right_hand_landmarks, connections.HAND_CONNECTIONS,
            drawer.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            drawer.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))