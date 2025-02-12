#ifndef EIGEN0CUDA.H
#define EIGEN0CUDA.H

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
/*
#include <stdio.h>
#include <stdlib.h>
*/

#define SQR(a) ((a)*(a))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a) )
 

void tridiag(double *a, int n, double *d, double *e);
 
//int eigstm(double *d, double *e, int n, double *z);
int eigstm(double *d, double *e, int n);

#endif
 
