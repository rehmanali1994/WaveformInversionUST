/*
 * This MEX file uses the mxGPUArray API with CUDA and CUBLAS to apply
 * the block LU decomposition of the 9-point Helmholtz equation.
 * 
 * This was initially implemented in MATLAB using the following code:
 * 
 * % Test if Block LU Decomposition Works
 * SRC = gpuArray(single(SRC));
 * u = complex(zeros(Ny, Nx, num_sources, 'single', 'gpuArray')); 
 * if adjHelmholtzEqn
 *     % Apply to Adjoint Helmholtz Solve
 *     for j = 2:Nx % Forward Pass
 *         u(:,j,:) = SRC(:,j,:) - triDiagMultLeftGPUmex(conj(Ud(:,j-1)), ...
 *             conj(Uu(:,j-1)), conj(Ul(:,j-1)), pagemtimes(invT(:,:,j-1)', u(:,j-1,:)));
 *     end; SRC = gather(SRC);
 *     for j = Nx-1:-1:1 % Backward Pass
 *         u(:,j,:) = pagemtimes(invT(:,:,j)', u(:,j,:) - ...
 *             triDiagMultLeftGPUmex(conj(Ld(:,j)), ...
 *             conj(Lu(:,j)), conj(Ll(:,j)), u(:,j+1,:)));
 *     end; u = gather(u);
 * else
 *     % Apply to Forward Helmholtz Solve
 *     for j = 2:Nx % Forward Pass
 *         u(:,j,:) = pagemtimes(invT(:,:,j), (SRC(:,j,:) - ...
 *             triDiagMultLeftGPUmex(Ld(:,j-1), Ll(:,j-1), Lu(:,j-1), u(:,j-1,:))));
 *     end; SRC = gather(SRC);
 *     for j = Nx-1:-1:1 % Backward Pass
 *         u(:,j,:) = u(:,j,:) - pagemtimes(invT(:,:,j), ...
 *             triDiagMultLeftGPUmex(Ud(:,j), Ul(:,j), Uu(:,j), u(:,j+1,:)));
 *     end; u = gather(u);  
 * end
 *
 * See the separate triDiagMultLeftGPUmex.cu file for additional details
 * regarding the algorithm above. Although triDiagMultLeftGPUmex.cu is not
 * used in this MEX file, key components of that algorithm were used here.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"
#include "typeinfo"
#include "cufft.h"
#include "math.h"
#include "matrix.h"
#include "cublas_v2.h" // To compile: mexcuda -lcublas applyBlockLU.cu

/* Helpful Operator Overloading -- Using cuComplex.h */
inline __host__ __device__ cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) {
    return cuCmulf(a,b);
}
inline __host__ __device__ void operator*=(cuFloatComplex &a, cuFloatComplex b) {
    const float2 c = a * b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) {
    return cuCdivf(a,b);
}
inline __host__ __device__ void operator/=(cuFloatComplex &a, cuFloatComplex b) {
    const float2 c = a / b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) {
    return cuCaddf(a,b);
}
inline __host__ __device__ void operator+=(cuFloatComplex &a, cuFloatComplex b) {
    const float2 c = a + b;
    a.x = c.x; a.y = c.y;
}
inline __host__ __device__ cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) {
    return cuCsubf(a,b);
}
inline __host__ __device__ void operator-=(cuFloatComplex &a, cuFloatComplex b) {
    const float2 c = a - b;
    a.x = c.x; a.y = c.y;
}

/* Constant Values */
const unsigned int threadsPerBlock1D = 32;
const cublasOperation_t noTranspose = CUBLAS_OP_N;
const cublasOperation_t transpose = CUBLAS_OP_T;
const cublasOperation_t conjugateTranspose = CUBLAS_OP_C;

/* Device code - Apply L Block */
void __global__ applyL(cuComplex const * __restrict__ SRC,
                       cuComplex const * __restrict__ Ld,
                       cuComplex const * __restrict__ Ll,
                       cuComplex const * __restrict__ Lu,
                       cuComplex const * __restrict__ u,
                       cuComplex * __restrict__ tmp, int x_idx, 
                       int const Nx, int const Ny,
                       int const numSources)
{   
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex su[threadsPerBlock1D][threadsPerBlock1D+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const src_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        /* Indexing */
        unsigned int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        // Regular Entries
        su[threadIdx.x][threadIdx.y+1] = u[row_idx - Ny];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (y_idx != 0) ) 
            su[threadIdx.x][threadIdx.y] = u[row_idx - Ny - 1];
        else if ( (threadIdx.y == blockDim.y-1) && (y_idx != Ny-1) )
            su[threadIdx.x][threadIdx.y+2] = u[row_idx - Ny + 1];
        __syncthreads();
        /* Tridiagonal Matrix Multiplication */
        if (y_idx == 0) {
            tmp[y_idx + Ny*src_idx] = SRC[row_idx] - 
                (Ld[y_idx + Ny*(x_idx-1)]*su[threadIdx.x][threadIdx.y+1] + 
                Lu[y_idx + (Ny-1)*(x_idx-1)]*su[threadIdx.x][threadIdx.y+2]); 
        } else if (y_idx == Ny-1) {
            tmp[y_idx + Ny*src_idx] = SRC[row_idx] - 
                (Ld[y_idx + Ny*(x_idx-1)]*su[threadIdx.x][threadIdx.y+1] + 
                Ll[y_idx-1 + (Ny-1)*(x_idx-1)]*su[threadIdx.x][threadIdx.y]); 
        } else {
            tmp[y_idx + Ny*src_idx] = SRC[row_idx] - 
                (Ld[y_idx + Ny*(x_idx-1)]*su[threadIdx.x][threadIdx.y+1] + 
                Ll[y_idx-1 + (Ny-1)*(x_idx-1)]*su[threadIdx.x][threadIdx.y] + 
                Lu[y_idx + (Ny-1)*(x_idx-1)]*su[threadIdx.x][threadIdx.y+2]); 
        } 
    }
}

/* cuBLAS code - Apply Inverse T Blocks (After L Blocks) */
void applyInvTpostL(cublasHandle_t handle,
                    cuComplex const * const invT,
                    cuComplex const * const tmp,
                    cuComplex * const u, int x_idx,
                    int const Nx, int const Ny,
                    int const numSources)
{   
    /* Scalars for cuBLAS GEMM call*/
    cuComplex alpha = make_cuFloatComplex(1,0);
    cuComplex beta = make_cuFloatComplex(0,0);
    
    /* Run cuBLAS GEMM code */
    cublasCgemm3m(handle, noTranspose, noTranspose,
                  Ny, numSources, Ny, 
                  &alpha, invT, Ny, tmp, Ny,
                  &beta, &u[Ny*x_idx], Nx*Ny); //u[y_idx + Ny*x_idx + Nx*Ny*src_idx] = res
}

/* Device code - Apply U Block */
void __global__ applyU(cuComplex const * __restrict__ Ud,
                       cuComplex const * __restrict__ Ul,
                       cuComplex const * __restrict__ Uu,
                       cuComplex const * __restrict__ u,
                       cuComplex * __restrict__ tmp, int x_idx, 
                       int const Nx, int const Ny,
                       int const numSources)
{   
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex su[threadsPerBlock1D][threadsPerBlock1D+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const src_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        /* Indexing */
        unsigned int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        // Regular Entries
        su[threadIdx.x][threadIdx.y+1] = u[row_idx + Ny];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (y_idx != 0) ) 
            su[threadIdx.x][threadIdx.y] = u[row_idx + Ny - 1];
        else if ( (threadIdx.y == blockDim.y-1) && (y_idx != Ny-1) )
            su[threadIdx.x][threadIdx.y+2] = u[row_idx + Ny + 1];
        __syncthreads();
        /* Tridiagonal Matrix Multiplication */
        if (y_idx == 0) {
            tmp[y_idx + Ny*src_idx] = 
                (Ud[y_idx + Ny*x_idx]*su[threadIdx.x][threadIdx.y+1] + 
                Uu[y_idx + (Ny-1)*x_idx]*su[threadIdx.x][threadIdx.y+2]); 
        } else if (y_idx == Ny-1) {
            tmp[y_idx + Ny*src_idx] = 
                (Ud[y_idx + Ny*x_idx]*su[threadIdx.x][threadIdx.y+1] + 
                Ul[y_idx-1 + (Ny-1)*x_idx]*su[threadIdx.x][threadIdx.y]); 
        } else {
            tmp[y_idx + Ny*src_idx] = 
                (Ud[y_idx + Ny*x_idx]*su[threadIdx.x][threadIdx.y+1] + 
                Ul[y_idx-1 + (Ny-1)*x_idx]*su[threadIdx.x][threadIdx.y] + 
                Uu[y_idx + (Ny-1)*x_idx]*su[threadIdx.x][threadIdx.y+2]); 
        } 
    }
}

/* cuBLAS code - Apply Inverse T Blocks (After U Blocks) */
void applyInvTpostU(cublasHandle_t handle,
                    cuComplex const * const invT,
                    cuComplex const * const tmp,
                    cuComplex * const u, int x_idx,
                    int const Nx, int const Ny,
                    int const numSources)
{   
    /* Scalars for cuBLAS GEMM call*/
    cuComplex alpha = make_cuFloatComplex(-1,0); // -= res
    cuComplex beta = make_cuFloatComplex(1,0); // -= res
    
    /* Run cuBLAS GEMM code */
    cublasCgemm3m(handle, noTranspose, noTranspose,
                  Ny, numSources, Ny, 
                  &alpha, invT, Ny, tmp, Ny,
                  &beta, &u[Ny*x_idx], Nx*Ny); //u[y_idx + Ny*x_idx + Nx*Ny*src_idx] -= res
}

/* Device code - Apply U Block (Adjoint) */
void __global__ applyUadj(cuComplex const * __restrict__ SRC,
                          cuComplex const * __restrict__ Ud,
                          cuComplex const * __restrict__ Ul,
                          cuComplex const * __restrict__ Uu,
                          cuComplex * __restrict__ u,
                          cuComplex const * __restrict__ tmp, int x_idx, 
                          int const Nx, int const Ny,
                          int const numSources)
{   
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex su[threadsPerBlock1D][threadsPerBlock1D+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const src_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        /* Indexing */
        unsigned int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        // Regular Entries
        su[threadIdx.x][threadIdx.y+1] = tmp[y_idx + Ny*src_idx];
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (y_idx != 0) ) 
            su[threadIdx.x][threadIdx.y] = tmp[y_idx + Ny*src_idx - 1];
        else if ( (threadIdx.y == blockDim.y-1) && (y_idx != Ny-1) )
            su[threadIdx.x][threadIdx.y+2] = tmp[y_idx + Ny*src_idx + 1];
        __syncthreads();
        /* Tridiagonal Matrix Multiplication */
        if (y_idx == 0) {
            u[row_idx] = SRC[row_idx] -
                (cuConjf(Ud[y_idx + Ny*(x_idx-1)])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Ul[y_idx + (Ny-1)*(x_idx-1)])*su[threadIdx.x][threadIdx.y+2]); 
        } else if (y_idx == Ny-1) {
            u[row_idx] = SRC[row_idx] -
                (cuConjf(Ud[y_idx + Ny*(x_idx-1)])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Uu[y_idx-1 + (Ny-1)*(x_idx-1)])*su[threadIdx.x][threadIdx.y]); 
        } else {
            u[row_idx] = SRC[row_idx] -
                (cuConjf(Ud[y_idx + Ny*(x_idx-1)])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Uu[y_idx-1 + (Ny-1)*(x_idx-1)])*su[threadIdx.x][threadIdx.y] + 
                cuConjf(Ul[y_idx + (Ny-1)*(x_idx-1)])*su[threadIdx.x][threadIdx.y+2]); 
        } 
    }
}

/* cuBLAS code - Apply Inverse T Blocks (Before U Adjoint Blocks) */
void applyInvTpreU(cublasHandle_t handle,
                   cuComplex const * const invT,
                   cuComplex * const tmp,
                   cuComplex const * const u, int x_idx,
                   int const Nx, int const Ny,
                   int const numSources)
{   
    /* Scalars for cuBLAS GEMM call*/
    cuComplex alpha = make_cuFloatComplex(1,0); 
    cuComplex beta = make_cuFloatComplex(0,0); 
    
    /* Run cuBLAS GEMM code */
    cublasCgemm3m(handle, conjugateTranspose, noTranspose,
                  Ny, numSources, Ny, 
                  &alpha, invT, Ny, &u[Ny*(x_idx-1)], Nx*Ny,
                  &beta, tmp, Ny); 
}


/* Device code - Apply L Block (Adjoint) */
void __global__ applyLadj(cuComplex const * __restrict__ Ld,
                          cuComplex const * __restrict__ Ll,
                          cuComplex const * __restrict__ Lu,
                          cuComplex const * __restrict__ u,
                          cuComplex * __restrict__ tmp, int x_idx, 
                          int const Nx, int const Ny,
                          int const numSources)
{   
    /* Shared memory for loading A from global memory */
    __shared__ cuComplex su[threadsPerBlock1D][threadsPerBlock1D+2];
    /* Calculate the global linear index, assuming a 1-d grid. */
    unsigned int const src_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int const y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        /* Indexing */
        unsigned int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        // Regular Entries
        su[threadIdx.x][threadIdx.y+1] = u[row_idx + Ny]; 
        // Entries at the Edge of Each Block of Threads
        if ( (threadIdx.y == 0) && (y_idx != 0) ) 
            su[threadIdx.x][threadIdx.y] = u[row_idx + Ny - 1]; 
        else if ( (threadIdx.y == blockDim.y-1) && (y_idx != Ny-1) )
            su[threadIdx.x][threadIdx.y+2] = u[row_idx + Ny + 1]; 
        __syncthreads();
        /* Tridiagonal Matrix Multiplication */
        if (y_idx == 0) {
            tmp[y_idx + Ny*src_idx] = u[row_idx] - 
                (cuConjf(Ld[y_idx + Ny*x_idx])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Ll[y_idx + (Ny-1)*x_idx])*su[threadIdx.x][threadIdx.y+2]); 
        } else if (y_idx == Ny-1) {
            tmp[y_idx + Ny*src_idx] = u[row_idx] -
                (cuConjf(Ld[y_idx + Ny*x_idx])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Lu[y_idx-1 + (Ny-1)*x_idx])*su[threadIdx.x][threadIdx.y]); 
        } else {
            tmp[y_idx + Ny*src_idx] = u[row_idx] -
                (cuConjf(Ld[y_idx + Ny*x_idx])*su[threadIdx.x][threadIdx.y+1] + 
                cuConjf(Lu[y_idx-1 + (Ny-1)*x_idx])*su[threadIdx.x][threadIdx.y] + 
                cuConjf(Ll[y_idx + (Ny-1)*x_idx])*su[threadIdx.x][threadIdx.y+2]); 
        } 
    }
}

/* cuBLAS code - Apply Inverse T Blocks (After L Adjoint Blocks) */
void applyInvTpostLadj(cublasHandle_t handle,
                       cuComplex const * const invT,
                       cuComplex const * const tmp,
                       cuComplex * const u, int x_idx,
                       int const Nx, int const Ny,
                       int const numSources)
{   
    /* Scalars for cuBLAS GEMM call*/
    cuComplex alpha = make_cuFloatComplex(1,0); 
    cuComplex beta = make_cuFloatComplex(0,0); 
    
    /* Run cuBLAS GEMM code */
    cublasCgemm3m(handle, conjugateTranspose, noTranspose,
                  Ny, numSources, Ny, 
                  &alpha, invT, Ny, tmp, Ny,
                  &beta, &u[Ny*x_idx], Nx*Ny); 
}



/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *SRC;
    mxGPUArray const *Ld;
    mxGPUArray const *Ll;
    mxGPUArray const *Lu;
    mxGPUArray const *Ud;
    mxGPUArray const *Ul;
    mxGPUArray const *Uu;
    mxGPUArray const *invT;
    bool adjHelmholtzEqn;
    mxGPUArray *u;
    mxGPUArray *tmp;
    cuComplex const *d_SRC;
    cuComplex const *d_Ld;
    cuComplex const *d_Ll;
    cuComplex const *d_Lu;
    cuComplex const *d_Ud;
    cuComplex const *d_Ul;
    cuComplex const *d_Uu;
    cuComplex const *d_invT;
    cuComplex *d_u;
    cuComplex *d_tmp;
    const mwSize *dims_SRC;
    int Nx, Ny, num_sources;

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlockX = threadsPerBlock1D;
    int const threadsPerBlockY = threadsPerBlock1D;

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
        !(mxIsLogicalScalar(prhs[8])) ) {
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Inputs 1-8 should be gpuArray. Input 9 should be a boolean.");
    }

    /* Assemble GPU Arrays from Inputs */
    SRC = mxGPUCreateFromMxArray(prhs[0]);
    Ld = mxGPUCreateFromMxArray(prhs[1]);
    Ll = mxGPUCreateFromMxArray(prhs[2]);
    Lu = mxGPUCreateFromMxArray(prhs[3]);
    Ud = mxGPUCreateFromMxArray(prhs[4]);
    Ul = mxGPUCreateFromMxArray(prhs[5]);
    Uu = mxGPUCreateFromMxArray(prhs[6]);
    invT = mxGPUCreateFromMxArray(prhs[7]);
    adjHelmholtzEqn = (bool)(*mxGetLogicals(prhs[8]));
    

    /* Error Checking with Array Dimensions */
    dims_SRC = mxGPUGetDimensions(SRC);
    Ny = (int) dims_SRC[0]; 
    Nx = (int) dims_SRC[1]; 
    num_sources = ((int)mxGPUGetNumberOfElements(SRC))/(Nx*Ny);
    if (mxGPUGetDimensions(Ld)[0] != Ny || mxGPUGetDimensions(Ld)[1] != Nx-1 || 
        mxGPUGetDimensions(Ll)[0] != Ny-1 || mxGPUGetDimensions(Ll)[1] != Nx-1 || 
        mxGPUGetDimensions(Lu)[0] != Ny-1 || mxGPUGetDimensions(Lu)[1] != Nx-1 || 
        mxGPUGetDimensions(Ud)[0] != Ny || mxGPUGetDimensions(Ud)[1] != Nx-1 || 
        mxGPUGetDimensions(Ul)[0] != Ny-1 || mxGPUGetDimensions(Ul)[1] != Nx-1 || 
        mxGPUGetDimensions(Uu)[0] != Ny-1 || mxGPUGetDimensions(Uu)[1] != Nx-1 ||
        mxGPUGetDimensions(invT)[0] != Ny || mxGPUGetDimensions(invT)[1] != Ny || mxGPUGetDimensions(invT)[2] != Nx) {
        printf("Ny = %d\n", Ny);
        printf("Nx = %d\n", Nx);
        printf("Number of Sources = %d\n", num_sources);
        printf("Number of Rows in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[0]);
        printf("Number of Columns in Ld = %d\n", (int)mxGPUGetDimensions(Ld)[1]);
        printf("Number of Rows in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[0]);
        printf("Number of Columns in Ll = %d\n", (int)mxGPUGetDimensions(Ll)[1]);
        printf("Number of Rows in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[0]);
        printf("Number of Columns in Lu = %d\n", (int)mxGPUGetDimensions(Lu)[1]);
        printf("Number of Rows in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[0]);
        printf("Number of Columns in Ud = %d\n", (int)mxGPUGetDimensions(Ud)[1]);
        printf("Number of Rows in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[0]);
        printf("Number of Columns in Ul = %d\n", (int)mxGPUGetDimensions(Ul)[1]);
        printf("Number of Rows in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[0]);
        printf("Number of Columns in Uu = %d\n", (int)mxGPUGetDimensions(Uu)[1]);
        printf("invT is %d x %d x %d\n", (int)mxGPUGetDimensions(invT)[0], 
                                         (int)mxGPUGetDimensions(invT)[1], 
                                         (int)mxGPUGetDimensions(invT)[2]);
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Input sizes incorrect");
    }

    /* Verify that each array is a single array before extracting the pointer. */
    if (mxGPUGetClassID(SRC) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Ld) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Ll) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Lu) != mxSINGLE_CLASS &&
        mxGPUGetClassID(Ud) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Ul) != mxSINGLE_CLASS && 
        mxGPUGetClassID(Uu) != mxSINGLE_CLASS &&
        mxGPUGetClassID(invT) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt("parallel:gpu:triDiagMultLeftGPUmex:InvalidInput", 
                          "Invalid input to triDiagMultLeftGPUmex: Inputs should all be single precision");
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_SRC = (cuComplex const *)(mxGPUGetDataReadOnly(SRC));
    d_Ld = (cuComplex const *)(mxGPUGetDataReadOnly(Ld));
    d_Ll = (cuComplex const *)(mxGPUGetDataReadOnly(Ll));
    d_Lu = (cuComplex const *)(mxGPUGetDataReadOnly(Lu));
    d_Ud = (cuComplex const *)(mxGPUGetDataReadOnly(Ud));
    d_Ul = (cuComplex const *)(mxGPUGetDataReadOnly(Ul));
    d_Uu = (cuComplex const *)(mxGPUGetDataReadOnly(Uu));
    d_invT = (cuComplex const *)(mxGPUGetDataReadOnly(invT));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    u = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(SRC),
                            mxGPUGetDimensions(SRC),
                            mxGPUGetClassID(SRC),
                            mxGPUGetComplexity(SRC),
                            MX_GPU_INITIALIZE_VALUES);
    d_u = (cuComplex *)(mxGPUGetData(u));

    /* Create a GPUArray to hold the intermediate result. */
    const mwSize dims[] = {(mwSize)Ny, (mwSize)num_sources};
    tmp = mxGPUCreateGPUArray(2, dims,
                              mxGPUGetClassID(SRC),
                              mxGPUGetComplexity(SRC),
                              MX_GPU_DO_NOT_INITIALIZE);
    d_tmp = (cuComplex *)(mxGPUGetData(tmp));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 2-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    dim3 dimBlockBP(threadsPerBlockX, threadsPerBlockY, 1);
	dim3 dimGridBP((num_sources + dimBlockBP.x - 1) / dimBlockBP.x,
		           (Ny + dimBlockBP.y - 1) / dimBlockBP.y, 1);

    /* Code to call cuBLAS */
    cublasHandle_t handle; // handle
    cublasCreate(&handle); // create handle -- must destroy later

    /* Whether applying forward or adjoint Helmholtz solver */
    if (adjHelmholtzEqn) { // Adjoint Helmholtz Solver
        /* Forward propagation on grid */
        for (int x_idx = 1; x_idx < Nx; x_idx++) {
            // Apply Block LU to Propagate Forward on Grid
            applyInvTpreU(handle, &d_invT[Ny*Ny*(x_idx-1)], d_tmp, d_u, x_idx, Nx, Ny, num_sources);
            applyUadj<<<dimGridBP, dimBlockBP>>>(d_SRC, d_Ud, d_Ul, d_Uu, 
                 d_u, d_tmp, x_idx, Nx, Ny, num_sources); 
        }
        /* Backward propagation on grid */
        for (int x_idx = Nx-2; x_idx >= 0; x_idx--) {
            // Apply Block LU to Propagate Backward on Grid
            applyLadj<<<dimGridBP, dimBlockBP>>>(d_Ld, d_Ll, d_Lu, 
                 d_u, d_tmp, x_idx, Nx, Ny, num_sources);
            applyInvTpostLadj(handle, &d_invT[Ny*Ny*x_idx], d_tmp, d_u, x_idx, Nx, Ny, num_sources);
        }
    } else { // Forward Helmholtz Solver
        /* Forward propagation on grid */
        for (int x_idx = 1; x_idx < Nx; x_idx++) {
            // Apply Block LU to Propagate Forward on Grid
            applyL<<<dimGridBP, dimBlockBP>>>(d_SRC, d_Ld, d_Ll, d_Lu, 
                 d_u, d_tmp, x_idx, Nx, Ny, num_sources); 
            applyInvTpostL(handle, &d_invT[Ny*Ny*x_idx], d_tmp, d_u, x_idx, Nx, Ny, num_sources);
        }
        /* Backward propagation on grid */
        for (int x_idx = Nx-2; x_idx >= 0; x_idx--) {
            // Apply Block LU to Propagate Backward on Grid
            applyU<<<dimGridBP, dimBlockBP>>>(d_Ud, d_Ul, d_Uu, 
                 d_u, d_tmp, x_idx, Nx, Ny, num_sources); 
            applyInvTpostU(handle, &d_invT[Ny*Ny*x_idx], d_tmp, d_u, x_idx, Nx, Ny, num_sources);
        }
    } 

    /* Must destroy cuBLAS handle before exiting */
    cublasDestroy(handle);
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(u);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(SRC);
    mxGPUDestroyGPUArray(Ld);
    mxGPUDestroyGPUArray(Ll);
    mxGPUDestroyGPUArray(Lu);
    mxGPUDestroyGPUArray(Ud);
    mxGPUDestroyGPUArray(Ul);
    mxGPUDestroyGPUArray(Uu);
    mxGPUDestroyGPUArray(invT);
    mxGPUDestroyGPUArray(u);
    mxGPUDestroyGPUArray(tmp);
}
