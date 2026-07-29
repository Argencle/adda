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
#include "const.h" // keep this first
#if !defined(OPENCL) || !defined(PRECISE_TIMING)
#	error "ocl_matvec_profile.c requires OPENCL and PRECISE_TIMING"
#endif
#include "ocl_matvec_profile.h" // corresponding header
// project headers
#include "io.h"
#include "vars.h"
// system headers
#include <stdlib.h>

typedef struct {
	cl_event first,last;
	enum precise_ocl_matvec_operation part;
} precise_ocl_matvec_event;

#define NO_OPEN_REGION ((size_t)-1)

static precise_ocl_matvec_event *events;
static size_t event_count,event_capacity,open_region=NO_OPEN_REGION;
static double profile[PROF_MV_OCL_PARTS];
static size_t profile_calls[PROF_MV_OCL_PARTS];

static const char *const operation_names[PROF_MV_OCL_PARTS]={
	[PROF_MV_OCL_CONJ_INPUT]             = "conjugate input",
	[PROF_MV_OCL_ZERO_X]                 = "zero X matrix",
	[PROF_MV_OCL_ARITH1]                 = "Arith1",
	[PROF_MV_OCL_ZERO_SLICES]            = "zero slices",
	[PROF_MV_OCL_ARITH2]                 = "Arith2",
	[PROF_MV_OCL_TRANSPOSE_YZ_FORWARD]   = "transpose YZ forward",
	[PROF_MV_OCL_ARITH3]                 = "Arith3",
	[PROF_MV_OCL_TRANSPOSE_YZ_BACKWARD]  = "transpose YZ backward",
	[PROF_MV_OCL_ARITH4]                 = "Arith4",
	[PROF_MV_OCL_ARITH5]                 = "Arith5",
	[PROF_MV_OCL_INPROD]                 = "inner-product kernel",
	[PROF_MV_OCL_CONJ_OUTPUT]            = "conjugate output",
	[PROF_MV_OCL_FFT_X_FORWARD]          = "FFT X forward",
	[PROF_MV_OCL_FFT_Z_FORWARD]          = "FFT Z forward",
	[PROF_MV_OCL_FFT_Y_FORWARD]          = "FFT Y forward",
	[PROF_MV_OCL_FFT_Y_BACKWARD]         = "FFT Y backward",
	[PROF_MV_OCL_FFT_Z_BACKWARD]         = "FFT Z backward",
	[PROF_MV_OCL_FFT_X_BACKWARD]         = "FFT X backward",
	[PROF_MV_OCL_UPLOAD_ARGUMENT]        = "upload argument",
	[PROF_MV_OCL_COPY_SURFACE]           = "surface buffer copy",
	[PROF_MV_OCL_DOWNLOAD_INPROD]        = "download inner product",
	[PROF_MV_OCL_DOWNLOAD_RESULT]        = "download result"
};

//======================================================================================================================

static void EnqueueFirstMatVecOpenCLProfileMarker(cl_event *const event)
// clEnqueueMarker is used instead of its OpenCL 1.2 replacement to retain the project's OpenCL 1.0 compatibility
{
	IGNORE_WARNING(-Wdeprecated-declarations);
	CL_CH_ERR(clEnqueueMarker(command_queue,event));
	STOP_IGNORE;
}

//======================================================================================================================

static precise_ocl_matvec_event *ReserveFirstMatVecOpenCLEvent(const enum precise_ocl_matvec_operation part)
// reserve storage for either one command event or the two marker events delimiting a command region
{
	precise_ocl_matvec_event *tmp;

	if (part<0 || part>=PROF_MV_OCL_PARTS)
		LogError(ALL_POS,"Invalid first-MatVec OpenCL profile category (%d)",(int)part);
	if (event_count==event_capacity) {
		const size_t new_capacity=event_capacity==0 ? 64 : 2*event_capacity;
		tmp=(precise_ocl_matvec_event *)realloc(events,new_capacity*sizeof(*events));
		if (tmp==NULL)
			LogError(ALL_POS,"Failed to allocate storage for %zu first-MatVec OpenCL profiling events",new_capacity);
		events=tmp;
		event_capacity=new_capacity;
	}
	events[event_count].first=NULL;
	events[event_count].last=NULL;
	events[event_count].part=part;
	return events+event_count++;
}

//======================================================================================================================

cl_event *FirstMatVecOpenCLProfileEvent(const enum precise_ocl_matvec_operation part)
// reserve an event that will be filled by one directly instrumented OpenCL command
{
	return &ReserveFirstMatVecOpenCLEvent(part)->first;
}

//======================================================================================================================

void BeginFirstMatVecOpenCLProfileRegion(const enum precise_ocl_matvec_operation part)
/* Enqueue a marker before a multi-command operation. Regions are used for clFFT because its output event may identify
 * only the last internal command, and the bundled Apple implementation does not populate that output event at all.
 */
{
	precise_ocl_matvec_event *event;

	if (open_region!=NO_OPEN_REGION)
		LogError(ALL_POS,"Nested first-MatVec OpenCL profiling regions are not supported");
	open_region=event_count;
	event=ReserveFirstMatVecOpenCLEvent(part);
	EnqueueFirstMatVecOpenCLProfileMarker(&event->first);
}

//======================================================================================================================

void EndFirstMatVecOpenCLProfileRegion(void)
// enqueue the marker delimiting the end of the current multi-command operation
{
	if (open_region==NO_OPEN_REGION)
		LogError(ALL_POS,"No first-MatVec OpenCL profiling region is active");
	EnqueueFirstMatVecOpenCLProfileMarker(&events[open_region].last);
	open_region=NO_OPEN_REGION;
}

//======================================================================================================================

static cl_ulong FirstMatVecOpenCLEventTimestamp(const cl_event event,const cl_profiling_info param,
	const size_t index)
// query one timestamp after verifying that the event is complete
{
	cl_int status;
	cl_ulong timestamp;

	if (event==NULL)
		LogError(ALL_POS,"OpenCL did not return first-MatVec profiling event #%zu",index);
	CL_CH_ERR(clGetEventInfo(event,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(status),&status,NULL));
	if (status!=CL_COMPLETE)
		LogError(ALL_POS,"First-MatVec OpenCL profiling event #%zu is not complete",index);
	CL_CH_ERR(clGetEventProfilingInfo(event,param,sizeof(timestamp),&timestamp,NULL));
	return timestamp;
}

//======================================================================================================================

void CollectFirstMatVecOpenCLProfile(void)
/* Collect only after the caller's existing blocking read or clFinish. This function deliberately performs no wait or
 * queue synchronization of its own.
 */
{
	if (open_region!=NO_OPEN_REGION)
		LogError(ALL_POS,"An OpenCL profiling region is still active at the end of the first MatVec");
	for (size_t i=0;i<event_count;i++) {
		cl_ulong start,end;
		const enum precise_ocl_matvec_operation part=events[i].part;
		const bool region=events[i].last!=NULL;

		if (region) {
			// exclude the marker commands themselves from the measured region
			start=FirstMatVecOpenCLEventTimestamp(events[i].first,CL_PROFILING_COMMAND_END,i);
			end=FirstMatVecOpenCLEventTimestamp(events[i].last,CL_PROFILING_COMMAND_START,i);
		}
		else {
			start=FirstMatVecOpenCLEventTimestamp(events[i].first,CL_PROFILING_COMMAND_START,i);
			end=FirstMatVecOpenCLEventTimestamp(events[i].first,CL_PROFILING_COMMAND_END,i);
		}
		if (end<start)
			LogError(ALL_POS,"Invalid timestamps for first-MatVec OpenCL profiling event #%zu",i);
		profile[part]+=(double)(end-start)*1E-9;
		profile_calls[part]++;
		CL_CH_ERR(clReleaseEvent(events[i].first));
		if (region) CL_CH_ERR(clReleaseEvent(events[i].last));
	}
	free(events);
	events=NULL;
	event_count=event_capacity=0;
}

//======================================================================================================================

static void PrintFirstMatVecOpenCLProfileGroup(const char *const name,const size_t first,const size_t last)
// print a group and only the operations that occurred in the captured MatVec
{
	size_t calls=0;
	double total=0;

	for (size_t i=first;i<last;i++) {
		total+=profile[i];
		calls+=profile_calls[i];
	}
	if (calls==0) return;
	PrintBoth(logfile,"  %-19s = "FFORMPT_OCL"\n",name,total);
	for (size_t i=first;i<last;i++) if (profile_calls[i]!=0)
		PrintBoth(logfile,"    %-25s "FFORMPT_OCL" (%zu calls)\n",operation_names[i],profile[i],profile_calls[i]);
}

//======================================================================================================================

void PrintFirstMatVecOpenCLProfile(void)
// print device timings separately from the host-observed elapsed time printed by oclmatvec.c
{
	PrintFirstMatVecOpenCLProfileGroup("kernel events",0,PROF_MV_OCL_KERNEL_END);
	PrintFirstMatVecOpenCLProfileGroup("FFT/transpose spans",PROF_MV_OCL_FFT_X_FORWARD,PROF_MV_OCL_REGION_END);
	PrintFirstMatVecOpenCLProfileGroup("transfers/copies",PROF_MV_OCL_UPLOAD_ARGUMENT,PROF_MV_OCL_PARTS);
}
