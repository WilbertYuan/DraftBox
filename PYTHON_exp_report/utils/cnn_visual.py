from keras import models
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Input
from keras._tf_keras.keras.utils import plot_model

# 定义模型
model2 = models.Sequential()
model2.add(Input(shape=(28, 28, 1)))
model2.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same', activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

# 绘制模型结构图
plot_model(
    model2,
    
    to_file='model2_lenet_style.png',  # 输出图片路径
    show_shapes=True,                  # 显示每层的张量形状
    show_layer_names=True,             # 显示每层名称
    dpi=300                            # 图片分辨率
)
