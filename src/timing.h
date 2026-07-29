/* Definitions for usual timing; should be completely portable
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
#ifndef __timing_h
#define __timing_h

// project headers
#include "os.h"
#include "parbas.h"
#ifdef SOLVER_LINALG_PROFILE
#	include <stddef.h>
#endif

#ifdef ADDA_MPI
#	define TIME_TYPE double
#	define GET_TIME() MPI_Wtime()
#elif defined(OPENCL)
#	define TIME_TYPE double
#	define GET_TIME() GetOpenCLWallTime()
double GetOpenCLWallTime(void);
#else
#	include <time.h>
#	define TIME_TYPE clock_t
#	define GET_TIME() clock()
#endif

#ifdef WINDOWS
#	include <windows.h> // all windows functions need this
#	define SYSTEM_TIME LARGE_INTEGER
#	define GET_SYSTEM_TIME(t) QueryPerformanceCounter(t)
#elif defined(POSIX)
/* It make sense to switch to clock_gettime, especially with types CLOCK_MONOTONIC or CLOCK_MONOTONIC_RAW, to be
 * independent of wall time synchronization, etc. (as is now for Windows functions). However, that would be not so
 * portable and is probably overkill.
 */
#	include <sys/time.h> // for timeval and gettimeofday
#	include <stdio.h>    // needed for definition of NULL
#	define SYSTEM_TIME struct timeval
// gettimeofday is described only in POSIX 1003.1-2001, but it should work for many other systems
#	define GET_SYSTEM_TIME(t) gettimeofday(t,NULL)
#else
#	include <time.h>
#	define SYSTEM_TIME time_t
#	define GET_SYSTEM_TIME(t) time(t)
#endif

void StartTime(void);
void InitTiming(void);
void FinalStatistics(void);
double DiffSystemTime(const SYSTEM_TIME * restrict t1,const SYSTEM_TIME * restrict t2);

#ifdef SOLVER_LINALG_PROFILE
enum solver_linalg_profile_function {
	PROF_LA_NINIT,
	PROF_LA_NCOPY,
	PROF_LA_NNORM2,
	PROF_LA_NDOTPROD,
	PROF_LA_NDOTPROD_CONJ,
	PROF_LA_NDOTPRODSELF_CONJ,
	PROF_LA_NDOTPRODSELF_CONJ_NORM2,
	PROF_LA_NINCREM110_CMPLX,
	PROF_LA_NINCREM011_CMPLX,
	PROF_LA_NINCREM110_D_C_CONJ,
	PROF_LA_NINCREM111_CMPLX,
	PROF_LA_NINCREM,
	PROF_LA_NDECREM,
	PROF_LA_NINCREM01,
	PROF_LA_NINCREM10,
	PROF_LA_NINCREM11_D_C,
	PROF_LA_NINCREM01_CMPLX,
	PROF_LA_NINCREM10_CMPLX,
	PROF_LA_NLINCOMB_CMPLX,
	PROF_LA_NLINCOMB1_CMPLX,
	PROF_LA_NLINCOMB1_CMPLX_CONJ,
	PROF_LA_NSUBTR,
	PROF_LA_NMULT,
	PROF_LA_NMULT_CMPLX,
	PROF_LA_NMULTSELF,
	PROF_LA_NMULTSELF_CONJ,
	PROF_LA_NMULTSELF_CMPLX,
	PROF_LA_NMULT_MAT,
	PROF_LA_NMULTSELF_MAT,
	PROF_LA_NCONJ,
	PROF_LA_COMM,
	PROF_LA_PARTS
};

extern TIME_TYPE SolverLinAlgProfileHost[PROF_LA_PARTS];
extern size_t SolverLinAlgProfileHostCalls[PROF_LA_PARTS];
extern TIME_TYPE SolverLinAlgProfileHostCurrentIter[PROF_LA_PARTS];
extern size_t SolverLinAlgProfileHostCurrentIterCalls[PROF_LA_PARTS];
extern int SolverLinAlgProfileActive,SolverLinAlgProfileIterationActive;

void BeginSolverLinAlgProfile(void);
void BeginSolverLinAlgProfileIteration(void);
void EndSolverLinAlgProfileIteration(int complete);
void EndSolverLinAlgProfile(void);
#endif

#endif // __timing_h
