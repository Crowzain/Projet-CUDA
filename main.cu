#include "eigen0Cuda.h"


void checkResult(double *hostEig, double *gpuEig, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    /*
    Eigen values matching test
    */
}

int main(int argc, char** argv){
    int rval            = 0;

    double * h_matrices = NULL;
    double * d_matrices = NULL;
    /*
        Reading matrices from a given file
        if (argc==1){
            fprintf(stderr, "not enough parameters");
            	rval=1;
        }
    */
    
   	int nMatrices = 100;
	int d_nMatrices = 2;
	int order = 12;

	int h_nBytes = order*nMatrices*sizeof(double);
    h_matrices = (double*)malloc(h_nBytes);

    int d_nBytes = order*d_nMatrices*sizeof(double);

	
	if (h_matrices){
        //set up device
        int dev = 0;
        cudaSetDevice(dev);

        //set up data size of vectors
        dim3 block(d_nMatrices);
        dim3 grid((nMatrices+block.x-1)/block.x);
        


        double* d_Matrices;
		cudaMalloc((double**)&d_Matrices, d_nBytes);

    }
    return rval;
}