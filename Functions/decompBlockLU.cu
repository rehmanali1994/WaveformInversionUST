/*
 * This MEX file uses the mxGPUArray API with CUDA and CUSOLVER to perform
 * the block LU decomposition of the 9-point Helmholtz equation.
 * 
 * This was initially implemented in MATLAB using the following code:
 * 
 * % Calculate the Schur Complements and Their Inverses
 * invT = complex(zeros(Ny, Ny, Nx, 'single', 'gpuArray'));
 * D = diag(Dd(:,1)) + diag(Dl(:,1),-1) + diag(Du(:,1),1); T = D; 
 * invTcurr = complex(zeros(Ny, 'single', 'gpuArray'));
 * for j = 2:Nx
 *     % Invert T over only the interior points
 *     invTcurr(2:end-1,2:end-1) = inv(T(2:end-1,2:end-1));
 *     invT(:,:,j-1) = invTcurr;
 *     D = diag(Dd(:,j)) + diag(Dl(:,j),-1) + diag(Du(:,j),1); 
 *     T = D - triDiagMultLeftGPUmex(Ld(:,j-1), Ll(:,j-1), Lu(:,j-1), ...
 *         triDiagMultRight(invT(:,:,j-1), Ud(:,j-1).', Ul(:,j-1).', Uu(:,j-1).'));
 * end
 * invT(:,:,Nx) = invInterior(T);
 * clearvars T D invTcurr; % Remove last T and invTcurr from memory
 *
 * See the separate triDiagMultRight.m and triDiagMultLeftGPUmex.cu file 
 * for additional details regarding the algorithm above. Although neither 
 * triDiagMultRight.m nor triDiagMultLeftGPUmex.cu are used in 
 * this MEX file, key components of those algorithms were used here.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "typeinfo"
#include "cufft.h"
#include "math.h"
#include "cusolverDn.h" // To compile: mexcuda -lcusolver decompBlockLU.cu

/* Helpful Operator Overloading -- Using cuComplex.h */
inline __host__ __device__ cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a,b);
}
inline __host__ __device__ void operator*=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a * b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) {
    return cuCdivf(a,b);
}
inline __host__ __device__ void operator/=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a / b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a,b);
}
inline __host__ __device__ void operator+=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a + b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) {
    return cuCsubf(a,b);
}
inline __host__ __device__ void operator-=(cuFloatComplex &a, cuFloatComplex b) {
    const cuFloatComplex c = a - b;
    a.x = c.x; a.y = c.y;
}

/* Choose a reasonably sized number of threads for the block. */
const unsigned int threadsPerBlockX = 32;
const unsigned int threadsPerBlockY = 32;

/* Device code - Create tridiagonal matrix */
void __global__ triDiagMat(cuComplex const * const Ad,
                           cuComplex const * const Al,
                           cuComplex const * const Au,
                           cuComplex * const A,
                           int const size)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < size && col_idx < size) {
        if (row_idx == col_idx) {
            A[row_idx + size*col_idx] = Ad[row_idx];
        } else if (row_idx == col_idx+1) {
            A[row_idx + size*col_idx] = Al[col_idx];
        } else if (row_idx == col_idx-1) {
            A[row_idx + size*col_idx] = Au[row_idx];
        } else {
            A[row_idx + size*col_idx] = make_cuFloatComplex(0,0);
        }
    }
}


/* Device code - Copy Interior of Matrix */
void __global__ copyInterior(cuComplex const * const A,
                             cuComplex * const B, int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        // Border
        if ( (row_idx == 0) || (row_idx == N-1) || 
             (col_idx == 0) || (col_idx == N-1) ) {
            B[row_idx + N*col_idx] = make_cuFloatComplex(0,0);
        } 
        // Interior
        else {
            B[row_idx + N*col_idx] = A[row_idx + N*col_idx];
        }
    }
}

/* Device code - Initialize Interior Identity Matrix */
void __global__ initializeIdentity(cuComplex * const A, int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        // Border
        if ( (row_idx != 0) && (row_idx != N-1) && (row_idx == col_idx) ) {
            A[row_idx + N*col_idx] = make_cuFloatComplex(1,0);
        } 
        // Interior
        else {
            A[row_idx + N*col_idx] = make_cuFloatComplex(0,0);
        }
    }
}

/* Device code - Tridiagonal Matrix Multiply from Left */
void __global__ triDiagMultLeftPlusD(cuComplex const * const A,
                                     cuComplex const * const Bd,
                                     cuComplex const * const Bl,
                                     cuComplex const * const Bu,
                                     cuComplex const * const Dd,
                                     cuComplex const * const Dl,
                                     cuComplex const * const Du,
                                     cuComplex * const C,
                                     int const size)
{
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex sA[threadsPerBlockY][threadsPerBlockX+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < size && col_idx < size) {
        // Regular Entries
        sA[threadIdx.y][threadIdx.x+1] = A[row_idx +size*col_idx];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.x == 0) && (row_idx != 0) ) 
            sA[threadIdx.y][threadIdx.x] = A[row_idx-1 + size*col_idx];
        else if ( (threadIdx.x == blockDim.x-1) && (row_idx != size-1) )
            sA[threadIdx.y][threadIdx.x+2] = A[row_idx+1 + size*col_idx];
        __syncthreads();
        // Additional Tri-Diagonal D Input
        cuComplex D = make_cuFloatComplex(0,0);
        if (row_idx < size && col_idx < size) {
            if (row_idx == col_idx) {
                D = Dd[row_idx];
            } else if (row_idx == col_idx+1) {
                D = Dl[col_idx];
            } else if (row_idx == col_idx-1) {
                D = Du[row_idx];
            } 
        }
        // Actual Tridiagonal Matrix Multiply
        if (row_idx == 0) {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] +
                Bu[row_idx]*sA[threadIdx.y][threadIdx.x+2]);
        } else if (row_idx == size-1) {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] +
                Bl[row_idx-1]*sA[threadIdx.y][threadIdx.x]);
        } else {
            C[row_idx + size*col_idx] = D - 
                (Bd[row_idx]*sA[threadIdx.y][threadIdx.x+1] + 
                Bl[row_idx-1]*sA[threadIdx.y][threadIdx.x] +
                Bu[row_idx]*sA[threadIdx.y][threadIdx.x+2]); 
        }
    }
}

/* Device code - Tridiagonal Matrix Multiply from Right */
void __global__ triDiagMultRight(cuComplex const * const A,
                                 cuComplex const * const Bd,
                                 cuComplex const * const Bl,
                                 cuComplex const * const Bu,
                                 cuComplex * const C,
                                 int const numRows,
                                 int const numCols)
{
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex sA[threadsPerBlockX][threadsPerBlockY+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < numRows && col_idx < numCols) {
        // Regular Entries
        sA[threadIdx.x][threadIdx.y+1] = A[row_idx + numRows*col_idx];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (col_idx != 0) ) 
            sA[threadIdx.x][threadIdx.y] = A[row_idx + numRows*(col_idx-1)];
        else if ( (threadIdx.y == blockDim.y-1) && (col_idx != numCols-1) )
            sA[threadIdx.x][threadIdx.y+2] = A[row_idx + numRows*(col_idx+1)];
        __syncthreads();
        // Actual Tridiagonal Matrix Multiply
        if (col_idx == 0) {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] +
                Bl[col_idx]*sA[threadIdx.x][threadIdx.y+2];
        } else if (col_idx == numCols-1) {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] +
                Bu[col_idx-1]*sA[threadIdx.x][threadIdx.y];
        } else {
            C[row_idx + numRows*col_idx] = 
                Bd[col_idx]*sA[threadIdx.x][threadIdx.y+1] + 
                Bu[col_idx-1]*sA[threadIdx.x][threadIdx.y] +
                Bl[col_idx]*sA[threadIdx.x][threadIdx.y+2]; 
        }
    }
}



/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *Ld;
    mxGPUArray const *Ll;
    mxGPUArray const *Lu;
    mxGPUArray const *Dd;
    mxGPUArray const *Dl;
    mxGPUArray const *Du;
    mxGPUArray const *Ud;
    mxGPUArray const *Ul;
    mxGPUArray const *Uu;
    mxGPUArray *T;
    mxGPUArray *Tinterior;
    mxGPUArray *invT;
    cuComplex const *d_Ld;
    cuComplex const *d_Ll;
    cuComplex const *d_Lu;
    cuComplex const *d_Dd;
    cuComplex const *d_Dl;
    cuComplex const *d_Du;
    cuComplex const *d_Ud;
    cuComplex const *d_Ul;
    cuComplex const *d_Uu;
    cuComplex *d_T;
    cuComplex *d_Tinterior;
    cuComplex *d_invT;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ( (nrhs!=9) || 
        !(mxIsGPUArray(prhs[0])) || 
        !(mxIsGPUArray(prhs[1])) || 
        !(mxIsGPUArray(prhs[2])) || 
        !(mxIsGPUArray(prhs[3])) || 
        !(mxIsGPUArray(prhs[4])) || 
        !(mxIsGPUArray(prhs[5])) || 
        !(mxIsGPUArray(prhs[6])) || 
        !(mxIsGPUArray(prhs[7])) || 
        !(mxIsGPUArray(prhs[8])) ) {
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Expecting 9 gpuArray inputs");
    }

    /* Assemble GPU Arrays from Inputs */
    Ld = mxGPUCreateFromMxArray(prhs[0]);
    Ll = mxGPUCreateFromMxArray(prhs[1]);
    Lu = mxGPUCreateFromMxArray(prhs[2]);
    Dd = mxGPUCreateFromMxArray(prhs[3]);
    Dl = mxGPUCreateFromMxArray(prhs[4]);
    Du = mxGPUCreateFromMxArray(prhs[5]);
    Ud = mxGPUCreateFromMxArray(prhs[6]);
    Ul = mxGPUCreateFromMxArray(prhs[7]);
    Uu = mxGPUCreateFromMxArray(prhs[8]);

    /* Error Checking with Array Dimensions */
    int Ny = (int) mxGPUGetDimensions(Dd)[0]; 
    int Nx = (int) mxGPUGetDimensions(Dd)[1]; 
    if (mxGPUGetDimensions(Ld)[0] != Ny || mxGPUGetDimensions(Ld)[1] != Nx-1 || 
        mxGPUGetDimensions(Ll)[0] != Ny-1 || mxGPUGetDimensions(Ll)[1] != Nx-1 || 
        mxGPUGetDimensions(Lu)[0] != Ny-1 || mxGPUGetDimensions(Lu)[1] != Nx-1 || 
        mxGPUGetDimensions(Dd)[0] != Ny || mxGPUGetDimensions(Dd)[1] != Nx || 
        mxGPUGetDimensions(Dl)[0] != Ny-1 || mxGPUGetDimensions(Dl)[1] != Nx || 
        mxGPUGetDimensions(Du)[0] != Ny-1 || mxGPUGetDimensions(Du)[1] != Nx || 
        mxGPUGetDimensions(Ud)[0] != Ny || mxGPUGetDimensions(Ud)[1] != Nx-1 || 
        mxGPUGetDimensions(Ul)[0] != Ny-1 || mxGPUGetDimensions(Ul)[1] != Nx-1 || 
        mxGPUGetDimensions(Uu)[0] != Ny-1 || mxGPUGetDimensions(Uu)[1] != Nx-1) {
        // Print Statements for Nx and Ny
        printf("Nx = %d\n", Nx);
        printf("Ny = %d\n", Ny);
        // Print Statements for Ld, Ll, and Lu
        printf("Number of Rows in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[0]);
        printf("Number of Columns in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[1]);
        printf("Number of Rows in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[0]);
        printf("Number of Columns in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[1]);
        printf("Number of Rows in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[0]);
        printf("Number of Columns in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[1]);
        // Print Statements for Dd, Dl, and Du
        printf("Number of Rows in Dd = %d\n", (int)mxGPUGetDimensions(Dd)[0]);
        printf("Number of Columns in Dd = %d\n", (int)mxGPUGetDimensions(Dd)[1]);
        printf("Number of Rows in Dl = %d\n", (int)mxGPUGetDimensions(Dl)[0]);
        printf("Number of Columns in Dl = %d\n", (int)mxGPUGetDimensions(Dl)[1]);
        printf("Number of Rows in Du = %d\n", (int)mxGPUGetDimensions(Du)[0]);
        printf("Number of Columns in Du = %d\n", (int)mxGPUGetDimensions(Du)[1]);
        // Print Statements for Ud, Ul, and Uu
        printf("Number of Rows in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[0]);
        printf("Number of Columns in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[1]);
        printf("Number of Rows in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[0]);
        printf("Number of Columns in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[1]);
        printf("Number of Rows in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[0]);
        printf("Number of Columns in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[1]);
        // Throw Error for Incorrect Array Sizes
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Input sizes incorrect");
    }

    /* Verify that A really is a single array before extracting the pointer. */
    if (mxGPUGetClassID(Ld) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Ll) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Lu) != mxSINGLE_CLASS &&
        mxGPUGetClassID(Dd) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Dl) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Du) != mxSINGLE_CLASS &&  
        mxGPUGetClassID(Ud) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Ul) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Uu) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Inputs should all be single precision");
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Ld = (cuComplex const *)(mxGPUGetDataReadOnly(Ld));
    d_Ll = (cuComplex const *)(mxGPUGetDataReadOnly(Ll));
    d_Lu = (cuComplex const *)(mxGPUGetDataReadOnly(Lu));
    d_Dd = (cuComplex const *)(mxGPUGetDataReadOnly(Dd));
    d_Dl = (cuComplex const *)(mxGPUGetDataReadOnly(Dl));
    d_Du = (cuComplex const *)(mxGPUGetDataReadOnly(Du));
    d_Ud = (cuComplex const *)(mxGPUGetDataReadOnly(Ud));
    d_Ul = (cuComplex const *)(mxGPUGetDataReadOnly(Ul));
    d_Uu = (cuComplex const *)(mxGPUGetDataReadOnly(Uu));

    /* Create GPUArrays to hold the intermediate results and get their underlying pointers. */
    mwSize dims_T[2] = {(mwSize) Ny, (mwSize) Ny};
    T = mxGPUCreateGPUArray(2, dims_T,
                            mxGPUGetClassID(Dd),
                            mxGPUGetComplexity(Dd),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_T = (cuComplex *)(mxGPUGetData(T));
    Tinterior = mxGPUCreateGPUArray(2, dims_T,
                                    mxGPUGetClassID(Dd),
                                    mxGPUGetComplexity(Dd),
                                    MX_GPU_DO_NOT_INITIALIZE);
    d_Tinterior = (cuComplex *)(mxGPUGetData(Tinterior));

    /* Create a GPUArray to hold the final result and get its underlying pointer. */
    mwSize dims_invT[3] = {(mwSize) Ny, (mwSize) Ny, (mwSize) Nx};
    invT = mxGPUCreateGPUArray(3, dims_invT,
                               mxGPUGetClassID(Dd),
                               mxGPUGetComplexity(Dd),
                               MX_GPU_DO_NOT_INITIALIZE);
    d_invT = (cuComplex *)(mxGPUGetData(invT));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 2-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    dim3 dimBlockBP(threadsPerBlockX, threadsPerBlockY, 1);
	dim3 dimGridBP((Ny + dimBlockBP.x - 1) / dimBlockBP.x,
		(Ny + dimBlockBP.y - 1) / dimBlockBP.y, 1);

    /* Prepare cuSOLVER for matrix inversion */
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH); // must create handle at the beginning
    int Lwork = 0; // size of work space: determine size of work space on next line
    cusolverDnCgetrf_bufferSize(cusolverH, Ny-2, Ny-2, 
                                &d_Tinterior[Ny+1], Ny, &Lwork);

    /* Prepare additional inputs for cuSOLVER: Workspace, devIpiv, devInfo */
    mwSize mwSizeLwork = (mwSize) Lwork;
    mxGPUArray * workspace = mxGPUCreateGPUArray(1, &mwSizeLwork,
                                                 mxGPUGetClassID(Dd),
                                                 mxGPUGetComplexity(Dd),
                                                 MX_GPU_DO_NOT_INITIALIZE);
    cuComplex * d_work = (cuComplex *)(mxGPUGetData(workspace));
    mwSize N = (mwSize) Ny;
    mxGPUArray * Ipiv = mxGPUCreateGPUArray(1, &N,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int * d_Ipiv = (int *)(mxGPUGetData(Ipiv)); // Assuming int is 32 bit
    mwSize one = 1;
    mxGPUArray * info = mxGPUCreateGPUArray(1, &one,
                                            mxINT32_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int * d_info = (int *)(mxGPUGetData(info)); // Assuming int is 32 bit

    /* Calculate the Schur Complements and Their Inverses */
    triDiagMat<<<dimGridBP, dimBlockBP>>>(d_Dd, d_Dl, d_Du, d_T, Ny);  
    for (int x_idx = 1; x_idx < Nx; x_idx++) {
        /* Invert Interior of T matrix */
        copyInterior<<<dimGridBP, dimBlockBP>>>(d_T, d_Tinterior, Ny);  
        initializeIdentity<<<dimGridBP, dimBlockBP>>>(&d_invT[Ny*Ny*(x_idx-1)], Ny);
        /* Run cuSOLVER to get inverses */
        cusolverDnCgetrf(cusolverH, Ny-2, Ny-2, 
                         &d_Tinterior[Ny+1], Ny, d_work, d_Ipiv, d_info);
        cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, Ny-2, Ny-2, /* nrhs */
                         &d_Tinterior[Ny+1], Ny, d_Ipiv, 
                         &d_invT[Ny*Ny*(x_idx-1)+Ny+1], Ny, d_info);
        /* Update Schur Complement */
        triDiagMultRight<<<dimGridBP, dimBlockBP>>>(&d_invT[Ny*Ny*(x_idx-1)], 
                                                    &d_Ud[Ny*(x_idx-1)], 
                                                    &d_Ul[(Ny-1)*(x_idx-1)], 
                                                    &d_Uu[(Ny-1)*(x_idx-1)], 
                                                    d_Tinterior, Ny, Ny);    
        triDiagMultLeftPlusD<<<dimGridBP, dimBlockBP>>>(d_Tinterior, 
                                                        &d_Ld[Ny*(x_idx-1)], 
                                                        &d_Ll[(Ny-1)*(x_idx-1)], 
                                                        &d_Lu[(Ny-1)*(x_idx-1)], 
                                                        &d_Dd[Ny*x_idx], 
                                                        &d_Dl[(Ny-1)*x_idx], 
                                                        &d_Du[(Ny-1)*x_idx], 
                                                        d_T, Ny);    
    }
    /* Invert Interior of T matrix */
    copyInterior<<<dimGridBP, dimBlockBP>>>(d_T, d_Tinterior, Ny);  
    initializeIdentity<<<dimGridBP, dimBlockBP>>>(&d_invT[Ny*Ny*(Nx-1)], Ny);
    /* Run cuSOLVER to get inverses */
    cusolverDnCgetrf(cusolverH, Ny-2, Ny-2, 
                     &d_Tinterior[Ny+1], Ny, d_work, d_Ipiv, d_info);
    cusolverDnCgetrs(cusolverH, CUBLAS_OP_N, Ny-2, Ny-2, /* nrhs */
                     &d_Tinterior[Ny+1], Ny, d_Ipiv, 
                     &d_invT[Ny*Ny*(Nx-1)+Ny+1], Ny, d_info);

    /* Must destroy cuSOLVER handle at the end */
    cusolverDnDestroy(cusolverH); 

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(invT);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(Ld);
    mxGPUDestroyGPUArray(Ll);
    mxGPUDestroyGPUArray(Lu);
    mxGPUDestroyGPUArray(Dd);
    mxGPUDestroyGPUArray(Dl);
    mxGPUDestroyGPUArray(Du);
    mxGPUDestroyGPUArray(Ud);
    mxGPUDestroyGPUArray(Ul);
    mxGPUDestroyGPUArray(Uu);
    mxGPUDestroyGPUArray(T);
    mxGPUDestroyGPUArray(Tinterior);
    mxGPUDestroyGPUArray(workspace);
    mxGPUDestroyGPUArray(Ipiv);
    mxGPUDestroyGPUArray(info);
    mxGPUDestroyGPUArray(invT);
}
