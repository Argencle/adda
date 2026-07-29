/* OpenCL event profiling for the first matrix-vector product
 *
 * Copyright (C) ADDA contributors
 * This file is part of ADDA.
 *
 * ADDA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * ADDA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ADDA. If not, see
 * <http://www.gnu.org/licenses/>.
 */
#ifndef __ocl_matvec_profile_h
#define __ocl_matvec_profile_h

#if defined(OPENCL) && defined(PRECISE_TIMING)

// project headers
#include "oclcore.h" // for cl_event

#define FFORMPT_OCL "%.6f" // event durations are reported in seconds with microsecond display precision

enum precise_ocl_matvec_operation {
	// kernels
	PROF_MV_OCL_CONJ_INPUT,
	PROF_MV_OCL_ZERO_X,
	PROF_MV_OCL_ARITH1,
	PROF_MV_OCL_ZERO_SLICES,
	PROF_MV_OCL_ARITH2,
	PROF_MV_OCL_ARITH3,
	PROF_MV_OCL_ARITH4,
	PROF_MV_OCL_ARITH5,
	PROF_MV_OCL_INPROD,
	PROF_MV_OCL_CONJ_OUTPUT,
	PROF_MV_OCL_KERNEL_END,
	// complete FFT and transpose regions, potentially containing several internal library commands
	PROF_MV_OCL_FFT_X_FORWARD=PROF_MV_OCL_KERNEL_END,
	PROF_MV_OCL_FFT_Z_FORWARD,
	PROF_MV_OCL_TRANSPOSE_YZ_FORWARD,
	PROF_MV_OCL_FFT_Y_FORWARD,
	PROF_MV_OCL_FFT_Y_BACKWARD,
	PROF_MV_OCL_TRANSPOSE_YZ_BACKWARD,
	PROF_MV_OCL_FFT_Z_BACKWARD,
	PROF_MV_OCL_FFT_X_BACKWARD,
	PROF_MV_OCL_REGION_END,
	// transfers and copies
	PROF_MV_OCL_UPLOAD_ARGUMENT=PROF_MV_OCL_REGION_END,
	PROF_MV_OCL_COPY_SURFACE,
	PROF_MV_OCL_DOWNLOAD_INPROD,
	PROF_MV_OCL_DOWNLOAD_RESULT,
	PROF_MV_OCL_PARTS
};

cl_event *FirstMatVecOpenCLProfileEvent(enum precise_ocl_matvec_operation part);
void BeginFirstMatVecOpenCLProfileRegion(enum precise_ocl_matvec_operation part);
void EndFirstMatVecOpenCLProfileRegion(void);
void CollectFirstMatVecOpenCLProfile(void);
void PrintFirstMatVecOpenCLProfile(void);

#	define PROFILE_FIRST_MV_OCL_EVENT(enabled,part) \
		((enabled) ? FirstMatVecOpenCLProfileEvent(part) : NULL)
#	define PROFILE_FIRST_MV_OCL_REGION_BEGIN(enabled,part) do { \
		if (enabled) BeginFirstMatVecOpenCLProfileRegion(part); \
	} while (0)
#	define PROFILE_FIRST_MV_OCL_REGION_END(enabled) do { \
		if (enabled) EndFirstMatVecOpenCLProfileRegion(); \
	} while (0)

#else

#	define PROFILE_FIRST_MV_OCL_EVENT(enabled,part) NULL
#	define PROFILE_FIRST_MV_OCL_REGION_BEGIN(enabled,part) ((void)0)
#	define PROFILE_FIRST_MV_OCL_REGION_END(enabled) ((void)0)

#endif

#endif // __ocl_matvec_profile_h
