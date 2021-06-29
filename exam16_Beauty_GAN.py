# 모듈 임포트!
import dlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# img = dlib.load_rgb_image('./imgs/12.jpg')
# plt.figure(figsize=(16,10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy()
# dets = detector(img) # 얼굴 좌표 만들기
# if len(dets) == 0:
#     print('no puede buscar las caras!\
#     cannot find faces!')
# else:
#     fig, ax = plt.subplots(1, figsize=(16, 10))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height() # 왼쪽 좌표, 위쪽 좌표, 너비 좌표, 높이 좌표
#         rect = patches.Rectangle((x,y), w,h, linewidth=2, # 사각형의 굵기
#                                  edgecolor='r', # 사각형 색깔
#                                  facecolor='none') # 사각형 안 채우기
#         ax.add_patch(rect)
#     ax.imshow(img_result)
#     plt.show()
#
# # 랜드마크 찾는 코드
# fig, ax = plt.subplots(1, figsize=(16,10))
# objs = dlib.full_object_detections()
# for detection in dets:
#     s = sp(img, detection) # 5개의 랜드마크 위치를 추적
#     objs.append(s) # 5개의 랜드마크 리스트에 추가
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3, # 지름
#                                 edgecolor='r', facecolor='r')
#         ax.add_patch(circle)
# ax.imshow(img_result)
# plt.show()
#
# # 얼굴만 잘라서 출력
# faces = dlib.get_face_chips(img, objs, size=256, padding=0.0)
# fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
# axes[0].imshow(img)
# for i, face in enumerate(faces):
#     axes[i+1].imshow(face)
# plt.show()

def align_faces(img, detector, sp): # 이미지에서 얼굴만 찾아내서 잘라내는 함수
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces

# detector 설정
detector = dlib.get_frontal_face_detector() # 이미지에서 얼굴만 찾아줌
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat') # 5개의 랜드마크를 이용하여 얼굴 찾는 모델

# test_img = dlib.load_rgb_image('./imgs/12.jpg')
# test_faces = align_faces(test_img, detector, sp)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
# axes[0].imshow(test_img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

def preprocess(img):
    return (img / 255. - 0.5) * 2
def deprocess(img):
    return (img + 1) / 2

img1 = dlib.load_rgb_image('./imgs/12.jpg')
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image('./imgs/makeup/vFG56.png')
img2_faces = align_faces(img2, detector, sp)

fig, axes = plt.subplots(1, 2, figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()