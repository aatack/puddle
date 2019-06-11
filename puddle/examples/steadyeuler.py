from random import uniform
import puddle.puddle as pd


layers = [((10), "tanh"), ((), "id")]
u_infinity = 5.0
back_pressure = 1.0


def parameterise_surface(t):
    """Return a point on the surface parameterised by t."""
    raise NotImplementedError()


x = pd.scalar()
y = pd.scalar()
boundary_layer = pd.vector(2)

u = pd.dependent([x, y], layers)
v = pd.dependent([x, y], layers)
rho = pd.dependent([x, y], layers)
p = pd.dependent([x, y], layers)

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
