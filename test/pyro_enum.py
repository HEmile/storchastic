import pyro
from pyro.infer import TraceEnum_ELBO, config_enumerate, infer_discrete
import pyro.distributions as dist
import torch

pyro.enable_validation()
pyro.set_rng_seed(0)
def model():
    z = pyro.sample("z", dist.Categorical(torch.ones(5, 5)))
    print('model z = {}'.format(z))

def guide():
    z = pyro.sample("z", dist.Categorical(torch.ones(5, 5)))
    print('guide z = {}'.format(z))

elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, config_enumerate(guide, "parallel"))