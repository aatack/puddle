from puddle.maths.wrapper import wrap_tf_function, ShapeFunctions
import tensorflow as tf


copy = ShapeFunctions.copy_first_shape

add = wrap_tf_function(tf.add, copy)
subtract = wrap_tf_function(tf.subtract, copy)
multiply = wrap_tf_function(tf.multiply, copy)
divide = wrap_tf_function(tf.divide, copy)

square = wrap_tf_function(tf.square, copy)
sqrt = wrap_tf_function(tf.sqrt, copy)
exp = wrap_tf_function(tf.exp, copy)

stack = wrap_tf_function(tf.stack, ShapeFunctions.stack_shapes)
