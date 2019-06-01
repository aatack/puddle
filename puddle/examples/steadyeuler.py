import puddle.puddle as pd


layers = [((10), "tanh"), ((), "id")]

x = pd.scalar()
y = pd.scalar()

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

trainer = pd.trainer()
trainer.add_sampler(pd.sampler.space([x, y], equations))
