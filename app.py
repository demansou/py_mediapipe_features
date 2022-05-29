from facial_recognition.face_detect import FaceDetect as fd
from face_mesh.face_mesh import MediaPipeDetection as face_mesh_mpd
from body_recognition.body_detect import MediaPipeDetection as body_mpd
from segmentation.segmentation import Segmentation as seg
import sys

if __name__ == '__main__':
    prog_req = sys.argv[1]

    if not prog_req:
        exit(0)
    if prog_req == "face":
        cascPath = './facial_recognition/static/haarcascade_frontalface_default.xml'
        fd.detect(cascPath)
    if prog_req == "face_mesh":
        face_mesh_mpd().detect()
    if prog_req == "body":
        body_mpd().detect()
    if prog_req == "segment":
        segment_bg = './segmentation/static/IMG_2077-cropped.jpg'
        seg().segment(segment_bg)