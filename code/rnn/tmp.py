import jax
import jax.numpy as np
from jax import grad, jit, vmap


a = np.arange(15).reshape(1, 3, 5)

print(a)
print(a[:, :-2])
print(a.shape[:-2])
print(*a.shape[:-2])
