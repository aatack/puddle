import tensorflow as tf


def glorot_weights(shape):
    """Generate a Xavier-initialised weight set of the given shape."""
    initialiser = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initialiser(shape))


def make_layer(input_shape, output_shape):
    """Create a function that applies a single network layer to an input."""
    weights = glorot_weights(input_shape[::-1] + output_shape)
    biases = glorot_weights(output_shape)

    def apply_layer(node):
        return tf.nn.sigmoid(
            tf.add(biases, tf.tensordot(node, weights, len(input_shape)))
        )

    return apply_layer


def compose(*operations):
    """Create a function that composes multiple layers."""

    def apply(node):
        current = node
        for operation in operations:
            current = operation(current)
        return current

    return apply
