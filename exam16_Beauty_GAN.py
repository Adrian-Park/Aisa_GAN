# 모듈 임포트!
import dlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# detector 설정
detector = dlib.get_frontal_face_detector() # 이미지에서 얼굴만 찾아줌
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat') # 5개의 랜드마크를 이용하여 얼굴 찾는 모델

img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

img_result = img.copy()
dets = detector(img) # 얼굴 좌표 만들기
if len(dets) == 0:
    print('no puede buscar las caras!\
    cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16, 10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height() # 왼쪽 좌표, 위쪽 좌표, 너비 좌표, 높이 좌표
        rect = patches.Rectangle((x,y), w,h, linewidth=2, # 사각형의 굵기
                                 edgecolor='r', # 사각형 색깔
                                 facecolor='none') # 사각형 안 채우기
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

# 랜드마크 찾는 코드
fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection) # 5개의 랜드마크 위치를 추적
    objs.append(s) # 5개의 랜드마크 리스트에 추가
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, # 지름
                                edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

# 얼굴만 잘라서 출력
faces = dlib.get_face_chips(img, objs, size=256, padding=0.0)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()