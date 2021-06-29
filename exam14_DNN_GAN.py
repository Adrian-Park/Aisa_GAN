# 모델 임포트
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GAN은 모델에 창의력을 부여
# ex) 한 장소의 여름 이미지를 주면 겨울 이미지를 생성
# https://m.blog.naver.com/euleekwon/221557899873
# https://zzsza.github.io/data/2017/12/27/gan/

# 이미지 저장 경로 및 각종 변수 세팅
OUT_DIR = './OUT_img/'
img_shape = (28, 28, 1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100

# 데이터 불러오기
(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)

# 데이터 설정
X_train = X_train / 127.5 - 1 # 데이터의 값은 -1 ~ 1까지
X_train = np.expand_dims(X_train, axis=3) # reshape. axis=3을 주어 차원을 하나 늘림
print(X_train.shape)

# build genderator 생성
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise)) # Dense층 추가, 100개 짜리 잡음 주기
generator_model.add(LeakyReLU(alpha=0.01)) # activation function은 LeakyReLU(마이너스 값을 조금 학습할 수 있는 형태)

generator_model.add(Dense(784, activation='tanh')) # 레이어 2축 쌓음
generator_model.add(Reshape(img_shape)) # 최종 출력 이미지 reshape
print(generator_model.summary())

# build discriminiator 생성
lrelu = LeakyReLU(alpha=0.01)
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape)) # 생성된 출력 이미지를 그려보기 위해 Flatten 작업

discriminator_model.add(Dense(128, activation=lrelu)) # activation function을 나타내는 방법1
# discriminator_model.add(LeakyReLU(alpha=0.01)) # activation function을 나타내는 방법2

discriminator_model.add(Dense(1, activation='sigmoid')) # 진품, 가품 분류의 이진분류이기 때문에 sigmoid 사용
print(discriminator_model.summary())

# 모델 compile
discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False # discriminator_model은 학습 X로 설정

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# real 이미지와 fake 이미지 생성
real = np.ones((batch_size,1)) # 1로 채워진 수 128개
print(real)
fake = np.zeros((batch_size, 1)) # 0으로 채워진 수 128개
print(fake)

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size) # X_train.shape[0]은 60000
    real_imgs = X_train[idx] # 128개 이미지

    z = np.random.normal(0, 1, (batch_size, noise)) # 평균 0 표준편차 1
    fake_imgs = generator_model.predict(z) # 128개 이미지

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real) # real 이미지를 학습시켰을 때의 loss 값
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake) # fake 이미지를 학습시켰을 때의 loss 값

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # 원본 이미지와 가짜 이미지 loss 값들의 평균
    discriminator_model.trainable = False # discriminator_model 학습 X

    z = np.random.normal(0, 1, (batch_size, noise)) # input은 랜덤하게 만들어진 잡음
    gan_hist = gan_model.train_on_batch(z, real) # z라는 한계치를 뽑고, batch에 대한 정답 라벨 real을 부여
    # train_on_batch와 fit의 차이점 : 큰 차이 X
    # train_on_batch의 경우, 넘겨 받은 데이터에 대해서 gradient vector를 계산해서 적용하고 끝내는 것이고(1epoch)
    # fit의 경우는 epoch과 batch_size를 한번에 모두 넘겨준다는 것 정도가 차이가 된다.
    # GAN의 경우, discriminator의 학습시 마다 generator가 생성하는 데이터가 변화하게 된다.
    # 즉 처음부터 모든 데이터가 존재하고 이를 한번에 학습시키는 fit과는 다르게, 한번씩 업데이트를 할때마다 모델이 변화하므로,
    # train_on_batch를 사용하는 것이 매우 합당함.

    if itr % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %(itr, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize = (row, col), sharey = True, sharex = True) # 4*4 크기 이미지
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off') # x축, y축 눈금 제거
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()