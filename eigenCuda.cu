#include "eigen0Cuda.h"
 

 /*
  ! compare double
 */
/*
 static int compdouble (void const *a, void const *b)
 {
    int ret = 0;
    double const *pa = a;
    double const *pb = b;
    double diff = *pa - *pb;
    if (diff > 0){
       ret = 1;
    }
    else if (diff < 0){
       ret = -1;
    }
    else{
       ret = 0;
    }

    return ret;
 }
*/

static int compdouble (const void * a, const void * b){
  return (*(double*)a > *(double*)b) ? 1 : (*(double*)a < *(double*)b) ? -1:0 ;
}
 
/* convert a symmetric matrix to tridiagonal form */
 

static double pythag(double a, double b)
{
	double absa, absb;
	absa = fabs(a);
	absb = fabs(b);
	if (absa > absb) return absa * sqrt(1.0 + SQR(absb / absa));
	else return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

__device__ void tridiag(double *a, int n, double *d, double *e)
{
	int    l, k, j, i;
	double scale, hh, h, g, f;

	for (i = n - 1; i > 0; i--) {
		l = i - 1;
		h = scale = 0.0;
		if (l > 0) {
			for (k = 0; k < l + 1; k++)
				scale += fabs(a[n * i + k]);
			if (scale == 0.0)
				e[i] = a[n * i + l];
			else {
				for (k = 0; k < l + 1; k++) {
					a[n * i + k] /= scale;
					h += a[n * i + k] * a[n * i + k];
				}
				f = a[n * i + l];
				g = (f >= 0.0 ? -sqrt(h) : sqrt(h));
				e[i] = scale * g;
				h -= f * g;
				a[n * i + l] = f - g;
				f = 0.0;
				for (j = 0; j < l + 1; j++) {
					/* Next statement can be omitted if eigenvectors not wanted */
					// a[n * j + i] = a[n * i + j] / h;
					g = 0.0;
					for (k = 0; k < j + 1; k++)
						g += a[n * j + k] * a[ n * i + k];
					for (k = j + 1; k < l + 1; k++)
						g += a[n * k + j] * a[n * i + k];
					e[j] = g / h;
					f += e[j] * a[n * i + j];
				}
				hh = f / (h + h);
				for (j = 0; j < l + 1; j++) {
					f = a[n * i + j];
					e[j] = g = e[j] - hh * f;
					for (k = 0; k < j + 1; k++)
						a[n * j + k] -= (f * e[k] + g * a[n * i + k]);
				}
			}
		} else
			e[i] = a[n * i + l];
		d[i] = h;
	}
	/* Next statement can be omitted if eigenvectors not wanted */
	// d[0] = 0.0;
	e[0] = 0.0;
	/* Contents of this loop can be omitted if eigenvectors not wanted except for statement d[i]=a[i][i]; */
	for (i = 0; i < n; i++) {
               /*
		l = i;
		if (d[i] != 0.0) {
			for (j = 0; j < l; j++) {
				g = 0.0;
				for (k = 0; k < l; k++)
					g += a[n * i + k] * a[n * k + j];
				for (k = 0; k < l; k++)
					a[n * k + j] -= g * a[n * k + i];
			}
		}
               */
		d[i] = a[n * i + i];
               /*
		a[n * i + i] = 1.0;
		for (j = 0; j < l; j++)
			a[n * j + i] = a[n * i + j] = 0.0;
               */
	}
}

/* calculate the eigenvalues and eigenvectors of a symmetric tridiagonal matrix */
//int eigstm(double *d, double *e, int n, double *z)


//d corresponds to the elements on the main diagonal
//e corresponds to the elements on the second diagonal
//n size of the matrix

__global__ void eigstm(double *d, double *e, int n)
{
	int     m, l, iter, i, k;
	double  s, r, p, g, f, dd, c, b;
	
	unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (i && i<n-1)
		e[i - 1] = e[i];
	e[n - 1] = 0.0;
	for (l = 0; l < n; l++) {
		iter = 0;
		do {
			for (m = l; m < n - 1; m++) {
				dd = fabs(d[m]) + fabs(d[m + 1]);
				if (fabs(e[m]) + dd == dd)
					break;
			}
			if (m != l) {
				if (iter++ == 30) return;
				g = (d[l + 1] - d[l]) / (2.0 * e[l]);
				r = pythag(g, 1.0);
				g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
				s = c = 1.0;
				p = 0.0;
				for (i = m - 1; i >= l; i--) {
					f = s * e[i];
					b = c * e[i];
					e[i + 1] = (r = pythag(f, g));
					if (r == 0.0) {
						d[i + 1] -= p;
						e[m] = 0.0;
						break;
					}
					s = f / r;
					c = g / r;
					g = d[i + 1] - p;
					r = (d[i] - g) * s + 2.0 * c * b;
					d[i + 1] = g + (p = s * r);
					g = c * r - b;
					/* Next loop can be omitted if eigenvectors not wanted */
                                        /*
					for (k = 0; k < n; k++) {
						f = z[n * k + i + 1];
						z[n * k + i + 1] = s * z[n * k + i] + c * f;
						z[n * k + i] = c * z[n * k + i] - s * f;
					}
                                        */
				}
				if (r == 0.0 && i >= l)
					continue;
				d[l] -= p;
				e[l] = g;
				e[m] = 0.0;
			}
		} while (m != l);
	}
	qsort(d,n,sizeof(double),compdouble);
        //return (0);
}

 
