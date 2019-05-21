import puddle.construction.space as _space
import puddle.construction.variable as _variable
import puddle.construction.builder as _builder
import puddle.construction.constant as _constant
import puddle.maths.wrapper as _wrapper
import puddle.maths.derivatives as _derivatives
import puddle.maths.common as _common
import puddle.maths.equation as _equation


space = _space.Space
scalar = _space.Scalar
vector = _space.Vector

variable = _variable.Variable
dependent = _variable.DependentVariable

builder = _builder.Builder

constant = _constant.Constant

derivative = _derivatives.Derivative
wrap_tf_function = _wrapper.wrap_tf_function
shape_functions = _wrapper.ShapeFunctions

add = _common.add
subtract = _common.subtract
multiply = _common.multiply
divide = _common.divide
square = _common.square
sqrt = _common.sqrt
exp = _common.exp

equate = _equation.Equation
