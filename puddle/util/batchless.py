import tensorflow as tf


def identity_function(x, name=None):
    """Mimics the identity function."""
    return x


activations = {
    "sigmoid": tf.nn.sigmoid,
    "relu": tf.nn.relu,
    "leaky-relu": tf.nn.leaky_relu,
    "tanh": tf.nn.tanh,
    "softmax": tf.nn.softmax,
    "id": identity_function,
}


def glorot_weights(shape):
    """Generate a Xavier-initialised weight set of the given shape."""
    initialiser = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initialiser(shape))


def make_layer(input_shape, output_shape, activation):
    """Create a function that applies a single network layer to an input."""
    input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
    output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

    weights = glorot_weights(input_shape[::-1] + output_shape)
    biases = glorot_weights(output_shape)

    def apply_layer(node):
        return activations[activation](
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
