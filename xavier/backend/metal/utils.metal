#include "utils.h"

uint strided_idx(uint id, constant const uint *ndim, constant const uint *shape, constant const int *stride)
{
    uint dim[MAX_NDIM];
    uint carry = id;
    for (int i = *ndim - 1; i >= 0; i--)
    {
        dim[i] = carry % shape[i];
        carry /= shape[i];
    }
    uint idx = 0;
    for (uint i = 0; i < *ndim; i++)
    {
        idx += dim[i] * stride[i];
    }
    return idx;
}