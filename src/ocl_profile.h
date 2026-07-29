/* OpenCL event profiling helpers
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
#ifndef __ocl_profile_h
#define __ocl_profile_h

#if defined(OCL_BLAS) && defined(SOLVER_LINALG_PROFILE)

// project headers
#include "oclcore.h" // for cl_event
#include "timing.h"  // for enum solver_linalg_profile_opencl_operation

/* Return storage for the event produced by the next profiled OpenCL command. The collector takes ownership of the
 * resulting event and releases it after reading its timestamps.
 */
cl_event *SolverLinAlgProfileOpenCLEvent(enum solver_linalg_profile_opencl_operation part);
void SolverLinAlgProfileCollectOpenCLEvents(void);
void SolverLinAlgProfileFreeOpenCLEvents(void);

#	define PROFILE_LA_OPENCL_EVENT(part) SolverLinAlgProfileOpenCLEvent(part)

#else

#	define PROFILE_LA_OPENCL_EVENT(part) NULL

#endif

#endif // __ocl_profile_h
