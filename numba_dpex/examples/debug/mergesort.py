import dpnp as np

import numba_dpex as ndpx


@ndpx.func(debug=True)
def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]

        # Into 2 halves
        R = arr[mid:]
        mergeSort(L)
        mergeSort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


@ndpx.kernel(debug=True)
def sort_array(unsorted):
    mergeSort(unsorted)


def main():
    # N = 5
    a = np.array([2, 5, 1, 7, 10])

    sort_array(a)
    print("sorted array", a)


if __name__ == "__main__":
    main()
