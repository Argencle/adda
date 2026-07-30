/* OpenCL backend for batched free-space far-field sums
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
#include "ocl_farfield.h" // corresponding header
// project headers
#include "cmplx.h"
#include "io.h"
#include "memory.h"
#include "oclcore.h"
#include "vars.h"

// defined and initialized in fft.c
extern size_t slicesize;

// Current view of the physical dipole polarization on the device. The view is valid only until the next solver run.
static cl_mem polarization_buffer;
static size_t polarization_offset;
static const doublecomplex *host_polarization;
static bool polarization_prepared;
static bool polarization_upload_required;

//======================================================================================================================

void PrepareOpenCLFarField(const doublecomplex *polarization)
// defer one upload of the host polarization until the first direct far-field batch
{
	host_polarization=polarization;
	polarization_buffer=bufargvec;
	polarization_offset=0;
	polarization_prepared=true;
	polarization_upload_required=true;
}

//======================================================================================================================

static void EnsureOpenCLPolarization(void)
// upload the selected host polarization only when the first direct far-field batch needs it
{
	if (!polarization_prepared) PrepareOpenCLFarField(pvec);
	if (polarization_upload_required) {
		CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufargvec,CL_FALSE,0,local_nRows*sizeof(doublecomplex),
			host_polarization,0,NULL,NULL));
		polarization_buffer=bufargvec;
		polarization_offset=0;
		polarization_upload_required=false;
	}
}

//======================================================================================================================

static size_t FarFieldBatchCapacity(const size_t count,const size_t phase_stride)
// cap a direction batch by the already allocated phase-table and result buffers
{
	size_t capacity;

	if (count==0) return 0;
	capacity=MIN(slicesize/phase_stride,local_nRows/3);
	return MIN(count,capacity);
}

//======================================================================================================================

static void FillDirectPhaseTables(doublecomplex * restrict phases,const double * restrict directions,
	const size_t first,const size_t count,const size_t phase_stride)
{
	size_t i;

	for (i=0;i<count;i++) {
		const double *n=directions+3*(first+i);
		doublecomplex *table=phases+i*phase_stride;

		imExp_arr(-kdX*n[0],boxX,table);
		imExp_arr(-kdY*n[1],boxY,table+boxX);
		imExp_arr(-kdZ*n[2],local_Nz_unif,table+boxX+boxY);
	}
}

//======================================================================================================================

bool CalcOpenCLFarFieldDirect(doublecomplex * restrict raw_sums,const double * restrict directions,const size_t count)
// calculate raw sum(P*exp(-ik*r.n)) for arbitrary free-space directions
{
	const size_t phase_stride=(size_t)boxX+(size_t)boxY+(size_t)local_Nz_unif;
	const size_t batch_capacity=FarFieldBatchCapacity(count,phase_stride);
	doublecomplex *phases;
	size_t batch,done=0,global_size,phase_values;

	if (count==0) return true;
	if (batch_capacity==0) return false;
	phase_values=MultOverflow(batch_capacity,phase_stride,ALL_POS_FUNC);
	MALLOC_VECTOR(phases,complex,phase_values,ALL);
	EnsureOpenCLPolarization();

	CL_CH_ERR(clSetKernelArg(clfarfield_direct,0,sizeof(cl_mem),&polarization_buffer));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,1,sizeof(size_t),&polarization_offset));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,2,sizeof(cl_mem),&bufposition));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,3,sizeof(cl_mem),&bufslices));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,4,sizeof(cl_mem),&bufresultvec));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,5,sizeof(size_t),&local_nvoid_Ndip));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,6,sizeof(phase_stride),&phase_stride));
	const size_t x_size=boxX,y_size=boxY;
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,7,sizeof(x_size),&x_size));
	CL_CH_ERR(clSetKernelArg(clfarfield_direct,8,sizeof(y_size),&y_size));

	while (done<count) {
		batch=MIN(batch_capacity,count-done);
		FillDirectPhaseTables(phases,directions,done,batch,phase_stride);
		CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufslices,CL_FALSE,0,
			batch*phase_stride*sizeof(doublecomplex),phases,0,NULL,NULL));
		global_size=batch*oclFarFieldWG;
		CL_CH_ERR(clEnqueueNDRangeKernel(command_queue,clfarfield_direct,1,NULL,&global_size,&oclFarFieldWG,0,NULL,NULL));
		CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufresultvec,CL_TRUE,0,3*batch*sizeof(doublecomplex),
			raw_sums+3*done,0,NULL,NULL));
		done+=batch;
	}
	Free_cVector(phases);
	return true;
}

//======================================================================================================================

static void ProjectedAxes(const int zero_axis,int * restrict fast_axis,int * restrict slow_axis,
	size_t * restrict fast_size,size_t * restrict slow_size,double * restrict kd_fast,double * restrict kd_slow)
{
	switch (zero_axis) {
		case 0:
			*fast_axis=1;
			*slow_axis=2;
			*fast_size=boxY;
			*slow_size=local_Nz_unif;
			*kd_fast=kdY;
			*kd_slow=kdZ;
			break;
		case 1:
			*fast_axis=0;
			*slow_axis=2;
			*fast_size=boxX;
			*slow_size=local_Nz_unif;
			*kd_fast=kdX;
			*kd_slow=kdZ;
			break;
		case 2:
			*fast_axis=0;
			*slow_axis=1;
			*fast_size=boxX;
			*slow_size=boxY;
			*kd_fast=kdX;
			*kd_slow=kdY;
			break;
		default: LogError(ONE_POS,"Invalid zero axis in ProjectedAxes");
	}
}

//======================================================================================================================

static void FillProjectedPhaseTables(doublecomplex * restrict phases,const double * restrict directions,
	const size_t first,const size_t count,const size_t phase_stride,const int fast_axis,const int slow_axis,
	const size_t fast_size,const size_t slow_size,const double kd_fast,const double kd_slow)
{
	size_t i;

	for (i=0;i<count;i++) {
		const double *n=directions+3*(first+i);
		doublecomplex *table=phases+i*phase_stride;

		imExp_arr(-kd_fast*n[fast_axis],fast_size,table);
		imExp_arr(-kd_slow*n[slow_axis],slow_size,table+fast_size);
	}
}

//======================================================================================================================

bool CalcOpenCLFarFieldProjected(doublecomplex * restrict raw_sums,const double * restrict directions,
	const size_t count,const doublecomplex * restrict projected,const size_t plane_size,const int zero_axis)
// evaluate an already projected dense 2D polarization grid for a batch of directions
{
	double kd_fast,kd_slow;
	doublecomplex *phases;
	size_t fast_size=0,slow_size=0,phase_stride,batch_capacity,batch,done=0,global_size,phase_values;
	int fast_axis=0,slow_axis=0;

	if (count==0) return true;
	ProjectedAxes(zero_axis,&fast_axis,&slow_axis,&fast_size,&slow_size,&kd_fast,&kd_slow);
	phase_stride=fast_size+slow_size;
	batch_capacity=FarFieldBatchCapacity(count,phase_stride);
	if (batch_capacity==0 || plane_size>local_nRows/3) return false;
	phase_values=MultOverflow(batch_capacity,phase_stride,ALL_POS_FUNC);
	MALLOC_VECTOR(phases,complex,phase_values,ALL);

	/* bufargvec is solver scratch after PrepareOpenCLFarField. If it was also the cached direct-polarization buffer,
	 * mark that cache dirty before overwriting it with the projected grid.
	 */
	CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufargvec,CL_FALSE,0,3*plane_size*sizeof(doublecomplex),projected,0,
		NULL,NULL));
	if (polarization_prepared && polarization_buffer==bufargvec) polarization_upload_required=true;

	CL_CH_ERR(clSetKernelArg(clfarfield_projected,0,sizeof(cl_mem),&bufargvec));
	CL_CH_ERR(clSetKernelArg(clfarfield_projected,1,sizeof(cl_mem),&bufslices));
	CL_CH_ERR(clSetKernelArg(clfarfield_projected,2,sizeof(cl_mem),&bufresultvec));
	CL_CH_ERR(clSetKernelArg(clfarfield_projected,3,sizeof(plane_size),&plane_size));
	CL_CH_ERR(clSetKernelArg(clfarfield_projected,4,sizeof(phase_stride),&phase_stride));
	CL_CH_ERR(clSetKernelArg(clfarfield_projected,5,sizeof(fast_size),&fast_size));

	while (done<count) {
		batch=MIN(batch_capacity,count-done);
		FillProjectedPhaseTables(phases,directions,done,batch,phase_stride,fast_axis,slow_axis,fast_size,slow_size,
			kd_fast,kd_slow);
		CL_CH_ERR(clEnqueueWriteBuffer(command_queue,bufslices,CL_FALSE,0,
			batch*phase_stride*sizeof(doublecomplex),phases,0,NULL,NULL));
		global_size=batch*oclFarFieldWG;
		CL_CH_ERR(clEnqueueNDRangeKernel(command_queue,clfarfield_projected,1,NULL,&global_size,&oclFarFieldWG,0,NULL,
			NULL));
		CL_CH_ERR(clEnqueueReadBuffer(command_queue,bufresultvec,CL_TRUE,0,3*batch*sizeof(doublecomplex),
			raw_sums+3*done,0,NULL,NULL));
		done+=batch;
	}
	Free_cVector(phases);
	return true;
}
