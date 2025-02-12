#include "eigen0Cuda.h"


void checkResult(double *hostEig, double *gpuEig, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    /*
    Eigen values matching test
    */
}

int main(int argc, char** argv){
    rval = 0;
    double * matrices = NULL;
    /*
        Reading matrices from a given file
        if (argc==1){
            fprintf(stderr, "not enough parameters");
            	rval=1;
        }
    */
   	int nMatrices = 1;
	int nMatricesGPU = 1;
	int order = 12;

	int nBytes = order*nMatrices*sizeof(double);
    matrices = (double*)malloc(nBytes);
    
	
	if (matrices){
        //set up device
        int dev = 0;
        cudaSetDevice(dev);

        //set up data size of vectors
        dim3 block       = nElem;
        dim3 grid        = (nElem/block);
        


        double* d_Matrices;
		cudaMalloc((double**)&d_Matrices, )
    }
}