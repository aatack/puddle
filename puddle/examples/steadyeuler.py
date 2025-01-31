from random import uniform
import puddle.puddle as pd


dense_layer = ((10,), "tanh")
id_layer = ((), "id")
relu_layer = ((), "relu")
u_infinity = 5.0
back_pressure = 0.0001


def parameterise_surface(t):
    """Return a point on the surface parameterised by t."""

    # A vertical wall in the middle of the flow
    return 0.5, 0.25 + 0.5 * t


def wrap_parameterised_surface():
    """Wrap a sample from the surface in the appropriate feed dict."""
    t = uniform(0, 1)
    _x, _y = parameterise_surface(t)
    return {x: _x, y: _y}


x = pd.scalar()
y = pd.scalar()

u = pd.dependent([x, y], [dense_layer, id_layer])
v = pd.dependent([x, y], [dense_layer, id_layer])
rho = pd.dependent([x, y], [dense_layer, relu_layer])
p = pd.dependent([x, y], [dense_layer, relu_layer])

rho_u = pd.multiply(rho, u)
rho_v = pd.multiply(rho, v)
rho_uu = pd.multiply(rho_u, u)
rho_vv = pd.multiply(rho_v, v)
rho_uv = pd.multiply(rho_u, v)

dpdx = pd.multiply(pd.derivative(p, x), pd.constant(-1))
dpdy = pd.multiply(pd.derivative(p, y), pd.constant(-1))

equations = [
    pd.equate(pd.add(pd.derivative(rho_u, x), pd.derivative(rho_v, y))),
    pd.equate(pd.add(pd.derivative(rho_uu, x), pd.derivative(rho_uv, y)), dpdx),
    pd.equate(pd.add(pd.derivative(rho_uv, x), pd.derivative(rho_vv, y)), dpdy),
]

upstream_boundary_conditions = [
    pd.equate(u, pd.constant(u_infinity)),
    pd.equate(v, pd.constant(0.0)),
]

side_boundary_condition = pd.equate(v, pd.constant(0.0))

no_slip_conditions = [pd.equate(u, pd.constant(0.0)), pd.equate(v, pd.constant(0.0))]

downstream_boundary_condition = pd.equate(p, pd.constant(back_pressure))

trainer = pd.trainer()
trainer.add_sampler(pd.sampler.space([x, y], equations))
trainer.add_sampler(
    pd.sampler.anonymous(
        [x, y],
        equations + upstream_boundary_conditions,
        lambda: {x: x.lower, y: uniform(y.lower, y.upper)},
        lambda: {
            equations[0]: 0.0,
            equations[1]: 0.0,
            equations[2]: 0.0,
            upstream_boundary_conditions[0]: 0.5,
            upstream_boundary_conditions[1]: 0.5,
        },
    ),
    weight=0.1,
)
trainer.add_sampler(
    pd.sampler.anonymous(
        [x, y],
        equations + [downstream_boundary_condition],
        lambda: {x: x.upper, y: uniform(y.lower, y.upper)},
        lambda: {
            equations[0]: 0.0,
            equations[1]: 0.0,
            equations[2]: 0.0,
            downstream_boundary_condition: 1.0,
        },
    ),
    weight=0.1,
)
trainer.add_sampler(
    pd.sampler.anonymous(
        [x, y],
        equations + no_slip_conditions,
        wrap_parameterised_surface,
        lambda: {
            equations[0]: 0.1,
            equations[1]: 0.1,
            equations[2]: 0.1,
            no_slip_conditions[0]: 0.35,
            no_slip_conditions[1]: 0.35,
        },
    ),
    weight=1.0,
)
trainer.add_sampler(
    pd.sampler.anonymous(
        [x, y],
        equations + [side_boundary_condition],
        lambda: {
            x: uniform(x.lower, x.upper),
            y: y.lower if uniform(0.0, 1.0) < 0.5 else y.upper,
        },
        lambda: {
            equations[0]: 0.2,
            equations[1]: 0.2,
            equations[2]: 0.2,
            side_boundary_condition: 0.4,
        },
    ),
    weight=0.1,
)
