import tensorflow as tf

def yolo_loss(y_true, y_pred, num_classes, num_anchors, lambda_coord=5.0, lambda_noobj=0.5):
    """
    YOLO Loss function.

    Args:
        y_true: Ground truth labels, shape (batch_size, grid_size, grid_size, num_anchors, num_classes + 5).
        y_pred: Predicted labels, shape (batch_size, grid_size, grid_size, num_anchors, num_classes + 5).
        num_classes: Number of object classes.
        num_anchors: Number of anchor boxes used in the YOLO model.
        lambda_coord: Weight for the localization loss.
        lambda_noobj: Weight for the confidence loss of non-object cells.

    Returns:
        Total YOLO loss.
    """
    # Split the y_pred tensor into its components (bounding box coordinates, objectness scores, and class probabilities)
    pred_xy = y_pred[..., :2]
    pred_wh = y_pred[..., 2:4]
    pred_confidence = y_pred[..., 4:5]
    pred_class_probs = y_pred[..., 5:]

    # Split the y_true tensor into its components (bounding box coordinates, objectness scores, and class labels)
    true_xy = y_true[..., :2]
    true_wh = y_true[..., 2:4]
    true_confidence = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    # Calculate the box coordinates for loss calculation
    pred_box_xy = tf.sigmoid(pred_xy)
    pred_box_wh = tf.exp(pred_wh)

    # Calculate the box coordinates of the ground truth
    true_box_xy = true_xy
    true_box_wh = tf.exp(true_wh)

    # Calculate the intersection and union areas for the IoU calculation
    intersect_wh = tf.maximum(tf.minimum(pred_box_wh / 2, true_box_wh / 2), 0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    true_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    iou_scores = intersect_area / (union_area + 1e-7)

    # Objectness loss
    obj_mask = true_confidence
    no_obj_mask = 1 - obj_mask
    obj_loss = obj_mask * tf.square(1 - pred_confidence) + lambda_noobj * no_obj_mask * tf.square(0 - pred_confidence)

    # Localization loss
    coord_loss = obj_mask * (lambda_coord * tf.reduce_sum(tf.square(true_box_xy - pred_box_xy), axis=-1) +
                             lambda_coord * tf.reduce_sum(tf.square(tf.sqrt(true_box_wh) - tf.sqrt(pred_box_wh)), axis=-1))

    # Class probability loss
    class_loss = obj_mask * tf.reduce_sum(tf.square(true_class_probs - pred_class_probs), axis=-1)

    # Total YOLO loss
    total_loss = tf.reduce_mean(coord_loss + obj_loss + class_loss)

    return total_loss



from yolo_loss import yolo_loss

# Assuming you have defined your YOLO model and loaded your dataset
model.compile(optimizer='adam', loss=yolo_loss(num_classes, num_anchors))
model.fit(x_train, y_train, epochs=..., batch_size=...)
