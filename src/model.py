from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
    Activation,
)

def conv_block(x, n_filters):
    x = Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(x, n_filters):
    x = conv_block(x, n_filters)
    p = MaxPooling2D(pool_size=(2, 2))(x)
    p = Dropout(rate=0.3)(p)
    return x, p


def decoder_block(x, skip_features, n_filters):
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, n_filters)
    return x


def unet_model(input_shape):

    input_shape = input_shape + (3,)
    inputs = Input(input_shape)

    # encoder downsampling (x4)
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # bottleneck
    b = conv_block(p4, 1024)

    # decoder upsampling (x4)
    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    outputs = Conv2D(filters=4, kernel_size=1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model