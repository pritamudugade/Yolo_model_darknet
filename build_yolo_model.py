import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape

def darknet53_block(inputs, filters):
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(filters * 2, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Add()([inputs, x])
    return x

def darknet53(inputs):
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = darknet53_block(x, filters=64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = darknet53_block(x, filters=128)
    x = darknet53_block(x, filters=128)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = darknet53_block(x, filters=256)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = darknet53_block(x, filters=512)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)
    x = darknet53_block(x, filters=1024)

    return x

def yolo_detection_head(inputs, num_classes, num_anchors):
    num_filters = (num_classes + 5) * num_anchors
    outputs = Conv2D(num_filters, (1, 1), activation='linear', padding='same')(inputs)
    outputs = Reshape((None, None, num_anchors, num_classes + 5))(outputs)
    return outputs

def yolo_loss(y_true, y_pred, num_classes, num_anchors, lambda_coord=5.0, lambda_noobj=0.5):
    # ... (same implementation as before)

# Complete YOLO model definition
def build_yolo_model(input_shape, num_classes, num_anchors):
    inputs = tf.keras.layers.Input(shape=input_shape)
    darknet_output = darknet53(inputs)
    detection_head_output = yolo_detection_head(darknet_output, num_classes, num_anchors)

    model = tf.keras.models.Model(inputs=inputs, outputs=detection_head_output)

    return model

# Example usage:
input_shape = (416, 416, 3)  # Replace with the desired input shape
num_classes = 80  # Replace with the number of object classes in your dataset
num_anchors = 5  # Replace with the number of anchor boxes used in the YOLO model

model = build_yolo_model(input_shape, num_classes, num_anchors)
model.compile(optimizer='adam', loss=yolo_loss(num_classes, num_anchors))
model.summary()
