from keras.models import *
from keras.layers import *
import os
import sys
current_path=os.getcwd()
sys.path.append(current_path)
from utils.fcos_det_preprocess import preprocess_
from utils.fcos_det_postprocess import postprocess_

class FcosDetRunner:
    def __init__(self,
                 model_path,
                 input_shape=(416, 416),
                 score_thres=0.5,
                 max_objects=100,
                 nms=True,
                 iou_thres=0.35,
                 e2e_coco=False):
        """
                detection function for ssd models.
                model_path: str, path to keras.model
                anchor_path: str, path to anchor file .npy

                image: str or np.ndarray or PIL.Image.Image
                input_shape: tuple(h, w)
                score_thres: float ~(0,1)
                only_max: bool
                iou_thres: float ~(0,1). Will be ignored when only_max is True
                """
        self.model = load_model(model_path)
        self.input_shape = input_shape
        assert (self.input_shape[0] == self.model.input_shape[1])
        self.score_thres = score_thres
        self.max_objects = max_objects
        self.iou_thres = iou_thres
        self.nms = nms
        self.e2e_coco = e2e_coco

    def run(self, image):
        """
        do inference on single image
        """

        img_data, scale, w_ori, h_ori = preprocess_(image, self.input_shape)
        
        outputs = self.model.predict(img_data)
        
        dets = postprocess_(outputs, self.max_objects, self.score_thres,
                            scale, self.input_shape, w_ori, h_ori, self.nms, self.iou_thres, e2e_coco= self.e2e_coco)
        return dets
