from samplers.random_sampler import RandomSampler
from samplers.stratified_sampler import StratifiedSampler
from samplers.coreset_sampler import CoresetSampler
from samplers.prototype_sampler import PrototypeSampler
from samplers.stratified_coreset import StratifiedCoreset

SAMPLERS = {
    "random": RandomSampler(),
    "stratified": StratifiedSampler(),
    "coreset": CoresetSampler(),
    "prototype": PrototypeSampler(),
    "stratified_coreset": StratifiedCoreset(),
}
