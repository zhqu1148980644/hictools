# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: infertypes=True
# cython: initializedcheck=False
# cython: language_level=3


import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.float cimport DBL_EPSILON
from libc.stdlib cimport malloc, free
np.import_array()


cdef extern from "./_tools.cpp" nogil:
    cdef struct LineIterator:
        int max_dim
        int max_coor[20]
        int coordinates[20]
        int strides[20]
        int back_strides[20]

    cdef struct ArrayInfo:
        int ndim
        int size
        int itemsize
        int shapes[20]
        int strides[20]

    cdef enum ExtendMode:
        NEAREST
        WRAP
        REFLECT
        MIRROR
        CONSTANT

    cdef struct ConvInfo:
        int *shift
        int size1
        int size2
        int length
        int nlines
        int step
        int symmetric
        int nonzero
        int numpoints
        ExtendMode mode

    void correlate1d_normal[T](const T *arrayp, T *output,
                               const T *k, T cval,
                               const ConvInfo *ci, LineIterator *liter)

    void correlate1d_points[T](const T *array, T *output,
                               const int *indexes, const T *k,
                               const ConvInfo *ci)

ctypedef fused DTYPE_t:
    char
    short
    int
    long
    float
    double


# cython version of goto_nextline used in correlate1d
cdef inline void goto_nextline(LineIterator * liter, DTYPE_t ** datap) nogil:
    cdef int i
    for i in range(liter.max_dim, -1, -1):
        if liter.coordinates[i] < liter.max_coor[i]:
            liter.coordinates[i] += 1
            datap[0] += liter.strides[i]
            break
        else:
            liter.coordinates[i] = 0
            datap[0] -= liter.back_strides[i]

cdef inline int symmetric_mode(DTYPE_t[:] weights, int filter_size,
                               int size1, int size2) nogil:
    cdef int i, symmetric = 1

    for i in range(1, (filter_size / 2) + 1):
        if fabs(weights[i + size1] - weights[size1 - i]) > DBL_EPSILON:
            symmetric = 0
            break
    if symmetric == 0:
        symmetric = -1
        for i in range(1, (filter_size / 2) + 1):
            if fabs(weights[size1 + i] + weights[size1 - i]) > DBL_EPSILON:
                symmetric = 0
                break

    return symmetric


cdef correlate1d(DTYPE_t * inputp,
                 DTYPE_t * outputp,
                 DTYPE_t[:] weights,
                 int axis,
                 int[:] indexes,
                 DTYPE_t cval,
                 ExtendMode mode,
                 int nonzero,
                 int size,
                 int[:, :] shape_info):

    cdef:
        DTYPE_t * k = <DTYPE_t *> &weights[0]
        ConvInfo ci
        ArrayInfo info
        LineIterator liter
        int * pindexes
        int i = 0, ii = 0, filter_size = weights.shape[0]

    info.size = size
    info.itemsize = sizeof(DTYPE_t)
    info.ndim = shape_info.shape[1]
    for i in range(info.ndim):
        info.shapes[i] = shape_info[0, i]
        info.strides[i] = shape_info[1, i]

    ci.mode = mode
    ci.nonzero = nonzero
    ci.length = info.shapes[axis]
    ci.nlines = size / ci.length
    ci.size1 = filter_size / 2
    ci.size2 = filter_size - ci.size1 - 1
    ci.step = info.strides[axis]
    ci.numpoints = 0 if indexes is None else indexes.shape[0]
    ci.symmetric = symmetric_mode(weights, filter_size, ci.size1, ci.size2)
    ci.shift = <int *> malloc((ci.size1 + ci.size2 + 1) * sizeof(int))
    ci.shift += ci.size1
    for i in range(-ci.size1, ci.size2 + 1):
        ci.shift[i] = ci.step * i
    ci.shift -= ci.size1

    if ci.numpoints == 0:
        liter.max_dim = info.ndim - 2
        for i in range(info.ndim):
            if i == axis:
                continue
            liter.max_coor[ii] = info.shapes[i] - 1
            liter.coordinates[ii] = 0
            liter.strides[ii] = info.strides[i]
            liter.back_strides[ii] = liter.strides[ii] * liter.max_coor[ii]
            ii += 1

    with nogil:
        if ci.numpoints == 0:
            correlate1d_normal(inputp, outputp, k, cval, &ci, &liter)
        else:
            pindexes = <int *> &indexes[0]
            correlate1d_points(inputp, outputp, pindexes, k, &ci)

    free(ci.shift)

def convolve1d(array,
               np.ndarray[DTYPE_t] weights,
               axis=-1,
               mode=4,
               cval=0.0,
               points=None,
               nonzero=False) -> np.ndarray:

    dtype = weights.dtype
    shape_info = np.array([np.array(array.shape),
                           np.array(array.strides) / array.itemsize]).astype(np.int32)
    if points is not None:
        if points.shape != array.ndim:
            raise ValueError('Number of row must equal to number of dimension of the input array.')
        indexes = (points * shape_info[1][:, None]).sum(axis=0).astype(np.int32)
        output_array = np.zeros(indexes.size, dtype=dtype)
    else:
        indexes = None
        output_array = np.zeros(array.shape, dtype=dtype)

        correlate1d[DTYPE_t](<DTYPE_t *>np.PyArray_DATA(array),
                             <DTYPE_t *>np.PyArray_DATA(output_array),
                             weights, range(array.ndim)[axis], indexes,
                             np.array(cval, dtype=dtype),
                             np.array(mode, dtype=np.int32),
                             nonzero, array.size, shape_info)

    return output_array




def convolve():
    pass
