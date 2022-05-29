import cv2
import mediapipe as mp

class MediaPipeDetection:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect(self, logging=False):
        cap = cv2.VideoCapture(0)
        with self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if logging:
                        print("Ignoring empty camera frame.")
                    continue
                image, results = self.__mediapipe_detection(frame, fm)
                if logging:
                    print(results)
                # self.__draw_landmarks(image, results)
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
        connections = self.mp_face_mesh
        default_tesselation_style = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        default_contours_style = self.mp_drawing_styles.get_default_face_mesh_contours_style()
        default_iris_style = self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        for face_landmarks in results.multi_face_landmarks:
            drawer.draw_landmarks(image, face_landmarks, connections.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, connection_drawing_spec=default_tesselation_style)
            drawer.draw_landmarks(image, face_landmarks, connections.FACEMESH_CONTOURS,
                landmark_drawing_spec=None, connection_drawing_spec=default_contours_style)
            drawer.draw_landmarks(image, face_landmarks, connections.FACEMESH_IRISES,
                landmark_drawing_spec=None, connection_drawing_spec=default_iris_style)