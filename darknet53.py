


def darknet53(inputs):
    # Initial 3x3 convolution with 32 filters and strides=1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Stage 1: 1 residual block followed by max pooling
    x = darknet53_block(x, filters=64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Stage 2: 2 residual blocks followed by max pooling
    x = darknet53_block(x, filters=128)
    x = darknet53_block(x, filters=128)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Stage 3: 8 residual blocks followed by max pooling
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Stage 4: 8 residual blocks followed by max pooling
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Stage 5: 4 residual blocks
    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)

    return x






