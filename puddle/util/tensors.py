import tensorflow as tf


def tensor_map(f, tensor, shape):
    """Map the values of a tensor, maintaining its shape."""
    if len(shape) == 0:
        return f(tensor)
    else:
        remaining_shape = shape[1:]
        subtensors = [
            tensor_map(f, tensor[i], remaining_shape) for i in range(shape[0])
        ]
        return tf.stack(subtensors, axis=0)
