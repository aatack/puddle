import puddle.construction.space as _space
import puddle.construction.variable as _variable
import puddle.construction.builder as _builder
import puddle.construction.constant as _constant
import puddle.construction.compiler as _compiler
import puddle.maths.wrapper as _wrapper
import puddle.maths.derivatives as _derivatives
import puddle.maths.common as _common
import puddle.maths.equation as _equation
import puddle.api.sampler as _sampler
import puddle.api.system as _system
import puddle.api.samplers.space as _space_sampler
import puddle.api.samplers.composite as _composite_sampler
import puddle.api.samplers.subspace as _subspace
import puddle.api.trainer as _trainer


space = _space.Space
scalar = _space.Scalar
vector = _space.Vector

variable = _variable.Variable
dependent = _variable.DependentVariable

builder = _builder.Builder

constant = _constant.Constant

compiler = _compiler.Compiler
compilation_data = _compiler.CompilationData

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

system = _system.System
sampler = _sampler.Sampler
sampler.space = _space_sampler.SpaceSampler
sampler.composite = _composite_sampler.CompositeSampler
sampler.hyperplane = _subspace.HyperplaneSampler

trainer = _trainer.Trainer
