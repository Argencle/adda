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
#include "const.h" // keep this first
#if !defined(OCL_BLAS) || !defined(SOLVER_LINALG_PROFILE)
#	error "ocl_profile.c requires OCL_BLAS and SOLVER_LINALG_PROFILE"
#endif
#include "ocl_profile.h" // corresponding header
// project headers
#include "io.h"
#include "timing.h"
// system headers
#include <stdlib.h>

typedef struct {
	cl_event event;
	enum solver_linalg_profile_opencl_operation part;
	int iteration;
} solver_linalg_profile_event;

static solver_linalg_profile_event *events;
static size_t event_count,event_capacity;

//======================================================================================================================

cl_event *SolverLinAlgProfileOpenCLEvent(const enum solver_linalg_profile_opencl_operation part)
// reserve one event record; the OpenCL enqueue call writes the event handle into the returned location
{
	solver_linalg_profile_event *tmp;

	if (!SolverLinAlgProfileActive) return NULL;
	if (part<0 || part>=PROF_LA_OPENCL_PARTS)
		LogError(ALL_POS,"Invalid OpenCL solver linear-algebra profile category (%d)",(int)part);
	if (event_count==event_capacity) {
		const size_t new_capacity=event_capacity==0 ? 64 : 2*event_capacity;
		if (new_capacity<event_capacity)
			LogError(ALL_POS,"Too many pending OpenCL solver linear-algebra profiling events");
		tmp=(solver_linalg_profile_event *)realloc(events,new_capacity*sizeof(*events));
		if (tmp==NULL)
			LogError(ALL_POS,"Failed to allocate storage for %zu OpenCL profiling events",new_capacity);
		events=tmp;
		event_capacity=new_capacity;
	}
	events[event_count].event=NULL;
	events[event_count].part=part;
	events[event_count].iteration=SolverLinAlgProfileIterationActive;
	return &events[event_count++].event;
}

//======================================================================================================================

void SolverLinAlgProfileCollectOpenCLEvents(void)
/* Collect commands only at locations where the solver already guarantees completion through a blocking read or
 * clFinish. Querying the status first makes an accidentally missing synchronization explicit without adding one.
 */
{
	size_t i;

	for (i=0;i<event_count;i++) {
		cl_int status;
		cl_ulong start,end;
		double elapsed;
		const enum solver_linalg_profile_opencl_operation part=events[i].part;

		if (events[i].event==NULL)
			LogError(ALL_POS,"OpenCL did not return solver linear-algebra profiling event #%zu",i);
		CL_CH_ERR(clGetEventInfo(events[i].event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL));
		if (status!=CL_COMPLETE)
			LogError(ALL_POS,"OpenCL solver linear-algebra profiling event #%zu is not complete",i);
		CL_CH_ERR(clGetEventProfilingInfo(events[i].event,CL_PROFILING_COMMAND_START,sizeof(start),&start,NULL));
		CL_CH_ERR(clGetEventProfilingInfo(events[i].event,CL_PROFILING_COMMAND_END,sizeof(end),&end,NULL));
		if (end<start)
			LogError(ALL_POS,"Invalid timestamps for OpenCL solver linear-algebra profiling event #%zu",i);
		/* This is the execution interval reported for the event returned by the enqueue API. In particular, clBLAS
		 * documents its output event as identifying a kernel execution instance; it is not assumed to include host
		 * enqueue overhead.
		 */
		elapsed=(double)(end-start)*1E-9;
		SolverLinAlgProfileOpenCL[part]+=elapsed;
		SolverLinAlgProfileOpenCLCalls[part]++;
		if (events[i].iteration) {
			SolverLinAlgProfileOpenCLCurrentIter[part]+=elapsed;
			SolverLinAlgProfileOpenCLCurrentIterCalls[part]++;
		}
		CL_CH_ERR(clReleaseEvent(events[i].event));
	}
	event_count=0;
}

//======================================================================================================================

void SolverLinAlgProfileFreeOpenCLEvents(void)
// release host storage after all event handles have been collected and released
{
	if (event_count!=0)
		LogError(ALL_POS,"%zu OpenCL solver linear-algebra profiling events were not collected",event_count);
	free(events);
	events=NULL;
	event_capacity=0;
}
