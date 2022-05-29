import cv2

class FaceDetect:
    def detect(cascPath):
        faceCascade = cv2.CascadeClassifier(cascPath)
        videoCapture = cv2.VideoCapture(0)

        while True:
            _, frame = videoCapture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray, 1.1, 9)
            
            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        videoCapture.release()
        cv2.destroyAllWindows()