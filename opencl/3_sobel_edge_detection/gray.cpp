#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <FreeImage.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_NUM     2

void Cleanup(cl_context ctx, cl_command_queue cmdq, cl_program prog,
             cl_kernel kern, cl_mem mem[MEM_NUM])
{
    for (int i = 0; i < MEM_NUM; i++) {
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

cl_kernel CreateKernel(cl_program program, cl_mem memObj[MEM_NUM])
{
    cl_kernel kernel = 0;
    cl_int errRet = 0;

    kernel = clCreateKernel(program, "gray", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create a kernel\n";
        return NULL;
    }

    errRet = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObj[0]);
    errRet |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObj[1]);
    if (errRet != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments\n";
        return NULL;
    }

    return kernel;
}

size_t RoundUp(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if (r == 0) {
        return globalSize;
    } else {
        return globalSize + groupSize - r;
    }
}

bool EnqueueKernel(cl_command_queue cmdq, cl_kernel kernel, int width, int height)
{
    size_t localWorkSize[2] = { 16, 16 };
    size_t globalWorkSize[2] = {RoundUp(localWorkSize[0], width),
                                RoundUp(localWorkSize[1], height)};
    cl_int errRet;

    errRet = clEnqueueNDRangeKernel(cmdq, kernel,
                                    2,      // work-dim
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

void ReadImageToMemory(cl_command_queue cmdq, cl_mem image, size_t width, size_t height, char **buf)
{
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};
    int errNum = 0;

    *buf = new char [width * height * 4];
    errNum = clEnqueueReadImage(cmdq,
                                image,          // cl_mem image
                                CL_TRUE,        // blocking read
                                origin,         // start
                                region,
                                0,              // row_pitch. 0, width * bytes_per_pixel
                                0,              // slice_pitch. 0, height * image_row_pitch
                                *buf,
                                0, NULL, NULL); // wait event and return event
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed to enqueue read image command!\n";
        delete (*buf);
    }
}

bool StoreImage(char *file, char *buf, int width, int height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(file);
    FIBITMAP *image = FreeImage_ConvertFromRawBits(
                                (BYTE*)buf,
                                width, height,
                                width * 4, 32,
                                0xFF000000, 0x00FF0000, 0x0000FF00);
    return FreeImage_Save(format, image, file);
}

cl_mem LoadImage(cl_context ctx, char *file, int &width, int &height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(file, 0);
    FIBITMAP *image = FreeImage_Load(format, file);

    // convert to 32bits
    FIBITMAP *tmp = image;
    image = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(tmp);

    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);

    // read to buf
    char *buf = new char[width * height * 4]; // 4 * 8 = 32 bits
    memcpy(buf, FreeImage_GetBits(image), width * height * 4);
    FreeImage_Unload(image);

    // create opencl image
    cl_image_format clFormat;
    clFormat.image_channel_order = CL_RGBA;
    clFormat.image_channel_data_type = CL_UNORM_INT8;

    cl_int errNum;
    cl_mem clImage;
    clImage = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              &clFormat,
                              width,
                              height,
                              0,
                              buf,
                              &errNum);
    if (errNum != CL_SUCCESS)
        return 0;
    return clImage;
}

cl_mem OutputImage(cl_context ctx, int width, int height)
{
    cl_image_format clFormat;
    clFormat.image_channel_order = CL_RGBA;
    clFormat.image_channel_data_type = CL_UNORM_INT8;

    cl_int errNum;
    cl_mem clImage;
    clImage = clCreateImage2D(ctx, CL_MEM_WRITE_ONLY,
                              &clFormat,
                              width,
                              height,
                              0,
                              NULL,
                              &errNum);
    if (errNum != CL_SUCCESS)
        return 0;
    return clImage;
}


int main(int argc, char **argv)
{
    cl_context ctx = 0;
    cl_command_queue cmdq = 0;
    cl_program program = 0;
    cl_device_id dev = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[MEM_NUM] = {0, 0};
    int width, height;
    cl_bool imageSupport = CL_FALSE;
    char *inputFile = NULL;
    char *outputFile = NULL;
    char *outputBuf = NULL;

    if (argc != 3) {
        std::cerr << argv[0] << " <Input Image> <Output Image>\n";
        exit(1);
    }
    inputFile = argv[1];
    outputFile = argv[2];


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

    // Check Image support for device
    clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
    if (imageSupport != CL_TRUE) {
        std::cerr << "The device doesn't support image!\n";
        goto err;
    }

    memObjects[0] = LoadImage(ctx, inputFile, width, height);
    if (memObjects[0] == 0) {
        std::cerr << "Failed to load the image!\n";
        goto err;
    }

    memObjects[1] = OutputImage(ctx, width, height);
    if (memObjects[0] == 0) {
        std::cerr << "Failed to create output image!\n";
        goto err;
    }

    // [4] create a program
    program = CreateProgram(ctx, dev, "gray.cl");
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
    if (!EnqueueKernel(cmdq, kernel, width, height)) {
        std::cerr << "Failed to enqueue a kernel!\n";
        goto err;
    }

    // read image back to system memory
    ReadImageToMemory(cmdq, memObjects[1], width , height, &outputBuf);
    if (!outputBuf) {
        std::cerr << "Failed to read image to system!\n";
        goto err;
    }

    // write image from system memory to output image file
    StoreImage(outputFile, outputBuf, width, height);

    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(ctx, cmdq, program, kernel, memObjects);
    return 0;

err:
    Cleanup(ctx, cmdq, program, kernel, memObjects);
    return EXIT_FAILURE;
}
