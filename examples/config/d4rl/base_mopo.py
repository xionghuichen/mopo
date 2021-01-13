from copy import deepcopy
from .base import base_params

mopo_params = deepcopy(base_params)
mopo_params['kwargs'].update({
    'separate_mean_var': False,
    'penalty_learned_var': True,
})