#include <iostream>
#include <fstream>
#include <sstream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


// A[M x P] * B[P x N] =  C[M x N]
//   i   k      k   j       i   j

#define M           2
#define P           3
#define N           2
#define SZ_A        (M * P)
#define SZ_B        (P * N)
#define SZ_C        (M * N)

//
//
//
// A
// | 1 2 3 |   | 2  -1 |   | 4  2 |
// | 3 2 1 | * | 1   0 | = | 8 -2 |
//             | 0   1 |
//
//
float A[ M * P ] = {
    1.0f, 2.0f, 3.0f,
    3.0f, 2.0f, 1.0f,
};
float B[ P * N ] = {
    2.0, -1.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
};

float C[ M * N ] = { 0.0f };

void PrintMatrix(char *name, float *mat, int row, int col)
{
    printf("%s:\n", name);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%10f ", mat[i * col + j]);
        }
        printf("\n");
    }
}

void InitData(void)
{
    int i, k, j;
    float tmp;

    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            tmp = 0.0f;
            for (k = 0; k < P; ++k) {
                tmp += A[i * P + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }

    PrintMatrix((char*)"A", A, M, P);
    PrintMatrix((char*)"B", B, P, N);
    PrintMatrix((char*)"C", C, M, N);
}

void Cleanup(cl_context ctx, cl_command_queue cmdq, cl_program prog,
             cl_kernel kern, cl_mem mem[3])
{
    for (int i = 0; i < 3; i++) {
        if (mem != 0)
            clReleaseMemObject(mem[i]);
    }
    if (cmdq != 0)
        clReleaseCommandQueue(cmdq);
    if (kern != 0)
        clReleaseKernel(kern);
    if (prog != 0)
        clReleaseProgram(prog);
    if (ctx != 0)
        clReleaseContext(ctx);
}

cl_context CreateContext(void)
{
    cl_int errRet;
    cl_platform_id platformId;
    cl_uint numPlatforms;
    cl_context ctx = NULL;

    // add only 1 platform and get it from platformId
    // the return numPlatforms can be checked if a platform is available.
    errRet = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (errRet != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Cannot find a platform!\n";
        return NULL;
    }

    // Create a context on GPU device, if no existing, try CPU
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformId,
        0
    };
    ctx = clCreateContextFromType(contextProperties,
                                      CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errRet);
    if (errRet != CL_SUCCESS) {
        std::cout << "Cannot create GPU context, try CPU...\n";
        ctx = clCreateContextFromType(contextProperties,
                                          CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errRet);
        if (errRet != CL_SUCCESS) {
            std::cerr << "Failed to create GPU or CPU context!\n";
            return NULL;
        }
    }
    return ctx;
}

cl_command_queue CreateCommandQueue(cl_context ctx, cl_device_id *dev)
{
    cl_int errRet;
    cl_device_id *devices;
    cl_command_queue cmdq = NULL;
    size_t bufSize = -1;

    // find a device
    errRet = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &bufSize);
    if (errRet != CL_SUCCESS || bufSize <= 0) {
        std::cerr << "Failed to get device info from context!\n";
        return NULL;
    }

    devices = new cl_device_id[bufSize / sizeof(cl_device_id)];
    errRet = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, bufSize, devices, NULL);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to get device ID\n";
        goto out;
    }

    // Create a command queue for the first device in context
    cmdq = clCreateCommandQueue(ctx, devices[0], 0, NULL);
    if (cmdq == NULL) {
        std::cerr << "Failed to create command queue\n";
        goto out;
    }

    // return the device
    *dev = devices[0];
out:
    delete [] devices;
    return cmdq;
}

cl_program CreateProgram(cl_context ctx, cl_device_id dev, const char *fileName)
{
    cl_int errRet;
    cl_program program;

    // read file from source to a C string buffer
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file " << fileName << "\n";
        return NULL;
    }
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();

    // create a program from the read buffer
    program = clCreateProgramWithSource(ctx, 1, (const char**)&srcStr, NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create a program from source!\n";
        return NULL;
    }

    // build the program
    errRet = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errRet != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        std::cerr << "Error in kernel: \n" << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

bool CreateMemObject(cl_context ctx, cl_mem memObj[3])
{
    memObj[0] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * SZ_A, A, NULL);
    memObj[1] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * SZ_B, B, NULL);
    memObj[2] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               sizeof(float) * SZ_C, NULL, NULL);
    if (memObj[0] == NULL || memObj[1] == NULL || memObj[2] == NULL)
        return false;

    return true;
}

cl_kernel CreateKernel(cl_program program, cl_mem memObj[3])
{
    cl_kernel kernel = 0;
    cl_int errRet = 0;
    cl_int m = M;
    cl_int n = N;
    cl_int p = P;

    kernel = clCreateKernel(program, "matrixMul", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create a kernel\n";
        return NULL;
    }

    errRet  = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    errRet |= clSetKernelArg(kernel, 1, sizeof(cl_int), &p);
    errRet |= clSetKernelArg(kernel, 2, sizeof(cl_int), &n);
    errRet |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObj[0]);
    errRet |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObj[1]);
    errRet |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObj[2]);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments\n";
        return NULL;
    }

    return kernel;
}

bool EnqueueKernel(cl_command_queue cmdq, cl_kernel kernel)
{
    size_t globalWorkSize[2] = { M, N };
    cl_int errRet;

    errRet = clEnqueueNDRangeKernel(cmdq, kernel,
                                    2,      // work-dim
                                    NULL,   // global work offset
                                    globalWorkSize,
                                    NULL,   // localWorkSize
                                    0,      // num of events in wait list
                                    NULL,   // event wait list
                                    NULL);  // event
    if (errRet != CL_SUCCESS)
        return false;

    return true;
}

bool VerifyResult(cl_command_queue cmdq, cl_mem memObj)
{
    cl_int errRet;
    float result[SZ_C];

    errRet = clEnqueueReadBuffer(cmdq, memObj, CL_TRUE, 0,
                                 SZ_C * sizeof(float), result,
                                 0, NULL, NULL);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to read memory object from device\n";
        return false;
    }

    PrintMatrix((char*)"result", result, M, N);

    for (int i = 0; i < SZ_C; i++) {
        if (result[i] != C[i])
            return false;
    }
    std::cout << std::endl;
    return true;
}

int main()
{
    cl_context ctx = 0;
    cl_command_queue cmdq = 0;
    cl_program program = 0;
    cl_device_id dev = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0, 0, 0};
    //cl_int errRet;

    InitData();

    // [1] create a context on the first platform for GPU(or try CPU)
    ctx = CreateContext();
    if (ctx == NULL) {
        std::cerr << "Failed to create a context!\n";
        exit(EXIT_FAILURE);
    }

    // [2] create a command queue
    cmdq = CreateCommandQueue(ctx, &dev);
    if (cmdq == NULL) {
        std::cerr << "Failed to create a command queue!\n";
        goto err;
    }

    // [3] create memory objects for data
    if (!CreateMemObject(ctx, memObjects)) {
        std::cerr << "Failed to create memory objects!\n";
        goto err;
    }

    // [4] create a program
    program = CreateProgram(ctx, dev, "matrixMul.cl");
    if (program == NULL) {
        std::cerr << "Failed to create a program!\n";
        goto err;
    }

    // [5] create kernel and set kernel parameters
    kernel = CreateKernel(program, memObjects);
    if (kernel == NULL) {
        std::cerr << "Failed to set up a kernel!\n";
        goto err;
    }

    // [6] enqueue the kernel for execution
    if (!EnqueueKernel(cmdq, kernel)) {
        std::cerr << "Failed to enqueue a kernel!\n";
        goto err;
    }

    // check the result
    if (VerifyResult(cmdq, memObjects[2])) {
        std::cout << "PASSED!\n";
    } else {
        std::cerr << "FAILED!\n";
        goto err;
    }

    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(ctx, cmdq, program, kernel, memObjects);
    return 0;

err:
    Cleanup(ctx, cmdq, program, kernel, memObjects);
    return EXIT_FAILURE;
}
