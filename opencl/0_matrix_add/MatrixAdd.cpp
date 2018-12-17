#include <iostream>
#include <fstream>
#include <sstream>

// use 1.2 API instead of 2.0 to avoid API warning
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define ARRAY_SIZE  128

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
    float a[ARRAY_SIZE], b[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (float)i;
        b[i] = 1.0;
    }

    memObj[0] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * ARRAY_SIZE, a, NULL);
    memObj[1] = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * ARRAY_SIZE, b, NULL);
    memObj[2] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               sizeof(float) * ARRAY_SIZE, NULL, NULL);
    if (memObj[0] == NULL || memObj[1] == NULL || memObj[2] == NULL)
        return false;

    return true;
}

cl_kernel CreateKernel(cl_program program, cl_mem memObj[3])
{
    cl_kernel kernel = 0;
    cl_int errRet = 0;

    kernel = clCreateKernel(program, "MatrixAdd", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create a kernel\n";
        return NULL;
    }

    errRet = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObj[0]);
    errRet |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObj[1]);
    errRet |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObj[2]);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments\n";
        return NULL;
    }

    return kernel;
}

bool EnqueueKernel(cl_command_queue cmdq, cl_kernel kernel)
{
    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 16 };
    cl_int errRet;

    errRet = clEnqueueNDRangeKernel(cmdq, kernel,
                                    1,      // work-dim
                                    NULL,   // global work offset
                                    globalWorkSize,
                                    localWorkSize,
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
    float result[ARRAY_SIZE];

    errRet = clEnqueueReadBuffer(cmdq, memObj, CL_TRUE, 0,
                                 ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to read memory object from device\n";
        return false;
    }

#ifdef _DEBUG
    for (int i = 0; i < ARRAY_SIZE; i++)
        std::cout << result[i] << " ";
#endif

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (result[i] != i + 1)
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
    program = CreateProgram(ctx, dev, "MatrixAdd.cl");
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
