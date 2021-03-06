# 모듈 임포트!
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# detector 설정
detector = dlib.get_frontal_face_detector() # 이미지에서 얼굴만 찾아줌
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

