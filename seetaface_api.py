from ctypes import *
import cv2
import matplotlib.pyplot as plt
import copy as cp

class Landmarks:
    def __init__(self):
        self.x = [0.0] * 5
        self.y = [0.0] * 5

class Face:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.landmarks = Landmarks()

class SeetaFace:
    def __init__(self):
        self.det_lib = 'seetaface/FaceDetection/build/libseeta_facedet_lib.so'
   	self.det_model = 'seetaface/FaceDetection/model/seeta_fd_frontal_v1.0.bin'
    	self.alg_lib = 'seetaface/FaceAlignment/build/libseeta_fa_lib.so'
    	self.alg_model = 'seetaface/FaceAlignment/model/seeta_fa_v1.1.bin'
    	self.vrf_lib = 'seetaface/FaceIdentification/build/libviplnet.so.4.5'
    	self.vrf_model = 'seetaface/FaceIdentification/model/seeta_fr_v1.0.bin'
        self.min_face_size = 40
        self.score_thresh = 2.0
        self.image_pyramid_scale_factor = 0.8
        self.window_step = [4, 4]

    def face_detect(self, img_path):
        lib = cdll.LoadLibrary(self.det_lib)
        class _Face(Structure):
                pass
        _Face._fields_ = [("null", c_bool), ("x", c_int), ("y", c_int), ("width", c_int), ("height", c_int), ("next", POINTER(_Face))]
        lib.detect.restype = POINTER(_Face)

        root = lib.detect(img_path, self.det_model, self.min_face_size, c_float(self.score_thresh),
                c_float(self.image_pyramid_scale_factor), self.window_step[0], self.window_step[1])
        faces = []
        face = Face()
        if root:
            while not root.contents.null:
                face.height = root.contents.height
                face.width = root.contents.width
                face.x = root.contents.x
                face.y = root.contents.y
                faces.append(cp.deepcopy(face))
                root = root.contents.next
        return faces

    def face_align(self, img_path):

        lib = cdll.LoadLibrary(self.alg_lib)

        class _Facelandmarks(Structure):
                pass
        _Facelandmarks._fields_ = [("null", c_bool), ("x", c_int * 5), ("y", c_int * 5), ("next", POINTER(_Facelandmarks))]
        lib.align.restype = POINTER(_Facelandmarks)

        root = lib.align(img_path, self.det_model, self.alg_model, self.min_face_size, c_float(self.score_thresh),
         c_float(self.image_pyramid_scale_factor), self.window_step[0], self.window_step[1])
        landmarks = []
        lm = Landmarks()
        if root:
            while not root.contents.null:
                for i in range(5):
                    lm.x[i] = root.contents.x[i]
                    lm.y[i] = root.contents.y[i]
                landmarks.append(cp.deepcopy(lm))
                root = root.contents.next
        return landmarks

    def face_verify(self, img_path1, img_path2):
        lib = cdll.LoadLibrary(self.vrf_lib)
        lib.verify.restype = c_float

        return lib.verify(img_path1, img_path2, self.det_model, self.alg_model, self.vrf_model, self.min_face_size,
         c_float(self.score_thresh), c_float(self.image_pyramid_scale_factor), self.window_step[0], self.window_step[1])
