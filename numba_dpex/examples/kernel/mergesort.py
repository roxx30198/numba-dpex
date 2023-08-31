import dpnp as np

import numba_dpex as nd

array3 = np.zeros(4)


@nd.kernel
def simple_add(array1, array2, array3):
    i = nd.get_global_id(0)
    array3[i] = array1[i] + array2[i]


N = 4
a = np.array([2, 4, 55, 66])

b = np.array([222, 1, 4, 34])
# array3 = np.zeros(N)
print(a.device, b.device)
for i in range(N):
    simple_add[nd.Range(N)](a, b, array3)

print(array3)
