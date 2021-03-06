# 导入panda，keras 和tensorflow
import pandas as pd
import tensorflow as tf
import keras
from keras import models, layers
# 加载样本数据集，划分为x和y DataFrame
df = pd.read_csv("https://github.com/bgweber/Twitch/raw/master/Recommendations/games-expand.csv")
x = df.drop(['label'], axis=1)
y = df['label']
# 定义Keras模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10,)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 使用自定义度量函数
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc
# 编译并拟合模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',
              metrics=[auc])
history = model.fit(x, y, epochs=100, batch_size=100,
                    validation_split = .2, verbose=0)

# 以H5格式保存模型
model.save("games.h5")