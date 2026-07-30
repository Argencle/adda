/* OpenCL backend for batched free-space far-field sums
 *
 * Copyright (C) ADDA contributors
 * This file is part of ADDA.
 */
#ifdef OPENCL

#ifndef __ocl_farfield_h
#define __ocl_farfield_h

// project headers
#include "types.h"
// system headers
#include <stdbool.h>
#include <stddef.h>

void PrepareOpenCLFarField(const doublecomplex *polarization);
bool CalcOpenCLFarFieldDirect(doublecomplex *raw_sums,const double *directions,size_t count);
bool CalcOpenCLFarFieldProjected(doublecomplex *raw_sums,const double *directions,size_t count,
	const doublecomplex *projected,size_t plane_size,int zero_axis);

#endif // __ocl_farfield_h

#endif // OPENCL
