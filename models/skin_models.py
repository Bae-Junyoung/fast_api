import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import phasepack
from datetime import datetime

class CheckIsSkin():
    def __init__(self, model_root):
        self.model = tf.keras.models.load_model(model_root)

    def __str__(self):
        return f'{self.model}'

    def train(self, image_dir):
        return

    def pred(self, image) -> int:
        img = np.float32(image)
        img_resize = cv2.resize(img, (224,224))
        test_dt = tf.data.Dataset.from_tensor_slices([img_resize])
        test_dt = test_dt.batch(1)
        test_dt = test_dt.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        pred = self.model.predict(test_dt)

        return 1 if pred[0][0] >= 0.5 else 0


class EvaluationModel():
    def __init__(self, model_root, user_id: int):
        self.user_id = user_id
        self.model = tf.keras.models.load_model(model_root)

    def __str__(self):
        return f'{self.model}'

    def train(self, image_dir):
        return

    def pred(self, img):
        output = self.model.predict(img)
        return output


class PoreModel():
    def __init__(self, model_root, user_id: int):
        self.user_id = user_id
        self.model = tf.keras.models.load_model(model_root)

    def __str__(self):
        return f'{self.model}'

    def train(self, image_dir):
        return

    def pred(self):
        return


class TextureModel():
    def __init__(self, user_id: int, img, side: str, time: datetime):
        self.user_id = user_id
        self.side = side
        self.time = time.strftime('%Y%m%d%H%M%S')
        self.file_name = f'{self.user_id}_texture_{self.side}_{self.time}.jpg'

        src = np.array(img).astype(np.uint8)
        src_max_pool = cv2.pyrDown(src, 2)
        hsv = cv2.cvtColor(src_max_pool, cv2.COLOR_RGB2HSV)
        hsv_lst = cv2.split(hsv)
        light_mask = cv2.threshold(hsv_lst[2], np.mean(hsv_lst[2]) + 2 * np.std(hsv_lst[2]), 255, cv2.THRESH_BINARY)[
            1].astype(np.uint8)
        light_mask_up_pool = cv2.pyrUp(light_mask.reshape(*light_mask.shape, 1))
        light_removed = cv2.inpaint(src, light_mask_up_pool, 1, cv2.INPAINT_TELEA)
        gray = cv2.cvtColor(light_removed, cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(phasepack.phasecong(gray)[0], np.quantile(phasepack.phasecong(gray)[0], 0.98), 1,
                             cv2.THRESH_BINARY)[1].astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        idx = []
        lst = []
        for i, x in enumerate(contours):
            if cv2.contourArea(x) >= 3:
                if cv2.arcLength(contours[i], True) >= 10:
                    idx.append(i)
                    lst.append(x)

        mask = np.zeros_like(gray)
        cnt = cv2.drawContours(mask, lst, -1, 255, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(mask, kernel, iterations=1)

        inpaint = cv2.inpaint(light_removed, dilation, 5, cv2.INPAINT_TELEA)
        inpaint_gray = cv2.cvtColor(inpaint, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
        sharp_clahe = clahe.apply(inpaint_gray)

        dist = cv2.distanceTransform(
            cv2.adaptiveThreshold(sharp_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 0),
            cv2.DIST_L2, 5)
        texture_output = \
        cv2.threshold(dist.astype(np.uint8), np.quantile(dist, 0.5), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        texture_score = (1 - (texture_output==255).sum() / (texture_output.shape[0]*texture_output.shape[1])) * 100

        cv2.imwrite(f'./static/images/{self.file_name}', texture_output)
        self.texture_score = int(texture_score)

    def __str__(self):
        return 'texture model'

    def get_img_url(self):
        return self.file_name

    def get_score(self):
        return self.texture_score



class WrinkleModel():
    def __init__(self, model_root, user_id: int):
        self.user_id = user_id
        self.model = tf.keras.models.load_model(model_root)

    def __str__(self):
        return f'{self.model}'

    def train(self, image_dir):
        return

    def pred(self):
        return


class TroubleModel():
    def __init__(self, model_root, user_id: int):
        self.user_id = user_id
        self.model = tf.keras.models.load_model(model_root)

    def __str__(self):
        return f'{self.model}'

    def train(self, image_dir):
        return

    def pred(self):
        return

