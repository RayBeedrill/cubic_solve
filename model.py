import tensorflow as tf
from keras.layers import Dense, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from utils import action_map_small

def acc(y_true, y_pred):
     return tf.cast(tf.equal(tf.argmax(y_true, axis=-1),
                            tf.argmax(y_pred, axis=-1)), tf.float32)


def get_model(lr=0.0001):
    input1 = Input((324,))

    d1 = Dense(1024)
    d2 = Dense(1024)
    d3 = Dense(1024)

    d4 = Dense(50)

    x1 = d1(input1)
    x1 = LeakyReLU()(x1)
    x1 = d2(x1)
    x1 = LeakyReLU()(x1)
    x1 = d3(x1)
    x1 = LeakyReLU()(x1)
    x1 = d4(x1)
    x1 = LeakyReLU()(x1)

    out_value = Dense(1, activation="linear", name="value")(x1)
    out_policy = Dense(len(action_map_small), activation="softmax", name="policy")(x1)

    model = Model(input1, [out_value, out_policy])

    model.compile(loss={"value": "mae", "policy": "sparse_categorical_crossentropy"}, optimizer=Adam(lr),
                  metrics={"policy": acc})
    model.summary()

    return model