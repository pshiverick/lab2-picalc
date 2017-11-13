#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef AOCL
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;
void cleanup();
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

#define NUM_ITEMS 4096

int main()
{
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_int ret;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    FILE *fp;
    char fileName[] = "./mykernel.cl";
    char *source_str;
    size_t source_size;

		float final_sum = 0.0;
		float pi = 0.0;
		const int num_wg = 16;
		const int num_wi = 16;
		int denominators[NUM_ITEMS];
		int i = 0;
		int value = 1;

		/* Populate array of values for denominators */
		for(i = 0; i < NUM_ITEMS; i++){
			denominators[i] = value;
			value += 2;
		}
	 
		float *result = (float *)malloc(sizeof(float)*num_wg);

#ifdef __APPLE__
    /* Get Platform and Device Info */
    clGetPlatformIDs(1, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    // we only use platform 0, even if there are more plantforms
    // Query the available OpenCL device.
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_LEN, dev_name, NULL);
    printf("device name= %s\n", dev_name);
#else

#ifdef AOCL  /* Altera FPGA */
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    // Get the OpenCL platform.
    platforms[0] = findPlatform("Intel(R) FPGA");
    if(platforms[0] == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }
    // Query the available OpenCL device.
    getDevices(platforms[0], CL_DEVICE_TYPE_ALL, &ret_num_devices);
    printf("Platform: %s\n", getPlatformName(platforms[0]).c_str());
    printf("Using one out of %d device(s)\n", ret_num_devices);
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("device name=  %s\n", getDeviceName(device_id).c_str());
#else
#error "unknown OpenCL SDK environment"
#endif

#endif

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
#ifdef __APPLE__
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
              (const size_t *)&source_size, &ret);
    if (ret != CL_SUCCESS) {
      printf("Failed to create program from source.\n");
      exit(1);
    }
#else

#ifdef AOCL  /* on FPGA we need to create kernel from binary */
   /* Create Kernel Program from the binary */
   std::string binary_file = getBoardBinaryFile("mykernel", device_id);
   printf("Using AOCX: %s\n", binary_file.c_str());
   program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);
#else
#error "unknown OpenCL SDK environment"
#endif

#endif

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
      printf("Failed to build program.\n");
			printf("ret = %d", ret);
      exit(1);
    }

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "calculatePi", &ret);
    if (ret != CL_SUCCESS) {
      printf("Failed to create kernel.\n");
      exit(1);
    }

		/* Create Buffers for I/O */
		cl_mem bufferDen = clCreateBuffer(context, CL_MEM_READ_ONLY,
				sizeof(int)*NUM_ITEMS, NULL, &ret);
	  clEnqueueWriteBuffer(command_queue, bufferDen, CL_TRUE, 0,
				sizeof(int)*NUM_ITEMS, (void *) denominators, 0, NULL, NULL);
		
		cl_mem resultBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
				sizeof(float)*num_wg, NULL, &ret);

    /* Execute the kernel */
		size_t globalws = (size_t)(num_wg*num_wi);
		size_t localws = (size_t)num_wi;

    /* Set the kernel arguments */
		/* Pass in Denominators */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferDen);
		/* Define size of local memory */
		ret |= clSetKernelArg(kernel, 1, sizeof(float)*num_wi, NULL);
		/* Pass in result buffer */
		ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&resultBuf);
		if(ret != CL_SUCCESS)
		{
			printf("Set kernel arguments failed\n");
			printf("return value: %d", ret);
		}

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
      &globalws, &localws, 0, NULL, NULL);
    /* it is important to check the return value.
      for example, when enqueueNDRangeKernel may fail when Work group size
      does not divide evenly into global work size */
    if (ret != CL_SUCCESS) {
      printf("Failed to enqueueNDRangeKernel.\n");
			printf("return value: %d\n", ret);
      exit(1);
    }

		/* Read results back -- array of floats that we must add up
		 * to get the final result */
		clEnqueueReadBuffer(command_queue, resultBuf, CL_TRUE, 0, 
				sizeof(float)*num_wg, (void *)result, 0, NULL, NULL);
		
		/* Add up result buffer and multiply by four to get the final result */
		final_sum = 0.0;
		for(i = 0; i < num_wg; i++)
		{
			final_sum += result[i];
		}
		pi = final_sum*4.0;

		/* Print out final result of pi that we calculated */
		printf("pi = %f\n", pi);

		free(result);

		clReleaseMemObject(resultBuf);
		clReleaseMemObject(bufferDen);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}

#ifdef AOCL
// Altera OpenCL needs this callback function implemented in main.c
// Free the resources allocated during initialization
void cleanup() {
}
#endif
