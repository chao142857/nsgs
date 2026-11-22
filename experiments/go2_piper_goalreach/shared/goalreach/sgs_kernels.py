"""The two fixed table kernels admitted by protocol K2."""
from types import MappingProxyType

KERNELS = MappingProxyType({
    'default': MappingProxyType(dict(history=100, target=.5, concentration=1., temperature=2., epsilon=1e-4)),
    'sharp': MappingProxyType(dict(history=100, target=.66, concentration=5., temperature=2., epsilon=1e-8)),
})
EMPTY_HISTORY_PRIOR = .5
