#pragma once

#include <metal_stdlib>

#define MAX_NDIM 8

uint access(uint id, constant const uint *ndim, constant const uint *shape, constant const int *stride);