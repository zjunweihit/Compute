#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#define check(msg, status)						\
do {									\
	if (status == HSA_STATUS_SUCCESS) {				\
		printf("%s: succeeded\n", #msg);			\
	} else {							\
		printf("%s: failed\n", #msg);				\
		exit(1);						\
	}								\
} while(0)

#define V_LEN 256

static float *in_A, *in_B, *out_C;

static hsa_status_t get_fine_region_cb(hsa_region_t region, void* data) {
	hsa_region_segment_t segment;
	assert(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment) == HSA_STATUS_SUCCESS);
	if (segment != HSA_REGION_SEGMENT_GLOBAL)
		return HSA_STATUS_SUCCESS;

	bool host_accessible_region = false;
	hsa_region_get_info(region, HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &host_accessible_region);
	hsa_region_global_flag_t flags;
	assert(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags) == HSA_STATUS_SUCCESS);
	if (host_accessible_region && (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED)) {
		hsa_region_t *p = (hsa_region_t*)data;
		*p = region;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}

static hsa_status_t check_gpu_agent_cb(hsa_agent_t agent, void *data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type == HSA_DEVICE_TYPE_GPU) {
		hsa_agent_t *p = (hsa_agent_t*)data;
		*p= agent;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}

// Actually we can print all regions associated with the agent for debug
static hsa_status_t check_kernarg_memory_region(hsa_region_t region, void *data)
{
	hsa_region_segment_t segment;
	assert(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment) == HSA_STATUS_SUCCESS);
	if (segment != HSA_REGION_SEGMENT_GLOBAL)
		return HSA_STATUS_SUCCESS;

	hsa_region_global_flag_t flags;
	assert(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags) == HSA_STATUS_SUCCESS);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
		hsa_region_t *p = (hsa_region_t*)data;
		*p = region;
		return HSA_STATUS_INFO_BREAK;
	}
	return HSA_STATUS_SUCCESS;
}

// not work??
//static void allocate_memory_host(void)
//{
//	uint32_t size = 256 * sizeof(float);
//	in_A = (float*)malloc(size);
//	in_B = (float*)malloc(size);
//	out_C = (float*)malloc(size);
//	assert(hsa_memory_register(in_A, size) == HSA_STATUS_SUCCESS);
//	assert(hsa_memory_register(in_B, size) == HSA_STATUS_SUCCESS);
//	assert(hsa_memory_register(out_C, size) == HSA_STATUS_SUCCESS);
//
//	for (int i = 0; i < 256; i++) {
//		in_A[i] = 1;
//		in_B[i] = 2;
//	}
//}

static void allocate_memory_fine(hsa_agent_t *agent)
{
	// Find host access fine region
	hsa_region_t region;
	region.handle=(uint64_t)-1;
	hsa_agent_iterate_regions(
			*agent, get_fine_region_cb,
			&region);
	assert(region.handle != (uint64_t)-1);

	assert(hsa_memory_allocate(region, V_LEN, (void **)&in_A) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_allocate(region, V_LEN, (void **)&in_B) == HSA_STATUS_SUCCESS);
	assert(hsa_memory_allocate(region, V_LEN, (void **)&out_C) == HSA_STATUS_SUCCESS);
	memset(out_C, 0, V_LEN);
	for (int i = 0; i < V_LEN; i++) {
		in_A[i] = i;
		in_B[i] = 1;
	}
}

static hsa_status_t init_packet(hsa_agent_t *agent,
				hsa_executable_t *executable,
				hsa_kernel_dispatch_packet_t *packet)
{
	packet->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
	packet->workgroup_size_x = (uint16_t)V_LEN;
	packet->workgroup_size_y = (uint16_t)1;
	packet->workgroup_size_z = (uint16_t)1;
	packet->grid_size_x = (uint32_t)V_LEN;
	packet->grid_size_y = (uint32_t)1;
	packet->grid_size_z = (uint32_t)1;

	// Extract info from symbol
	hsa_executable_symbol_t symbol;
	assert(hsa_executable_get_symbol_by_name(
			*executable, "vector_add", agent,
			&symbol) == HSA_STATUS_SUCCESS);
	assert(hsa_executable_symbol_get_info(
			symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
			&packet->kernel_object) == HSA_STATUS_SUCCESS);
	assert(hsa_executable_symbol_get_info(
			symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
			&packet->private_segment_size) == HSA_STATUS_SUCCESS);
	assert(hsa_executable_symbol_get_info(
			symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
			&packet->group_segment_size) == HSA_STATUS_SUCCESS);

	// Find kernel argument region
	hsa_region_t kernarg_region;
	kernarg_region.handle=(uint64_t)-1;
	hsa_agent_iterate_regions(
			*agent, check_kernarg_memory_region,
			&kernarg_region);
	assert(kernarg_region.handle != (uint64_t)-1);

	// allocate_memory_host(); // not work??
	allocate_memory_fine(agent);

	// Allocate kernel argument buffer from kernarg region
	// argument size can be got from executable symbol info
	struct kern_args_t {
		float *in_A;
		float *in_B;
		float *out_C;
	} *args = NULL;
	assert(hsa_memory_allocate(kernarg_region, sizeof(args),
			(void**)&args) == HSA_STATUS_SUCCESS);
	args->in_A = in_A;
	args->in_B = in_B;
	args->out_C = out_C;

	packet->kernarg_address = (void*)args;
	return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
	hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

	// Find gpu agent
	hsa_agent_t gpu_agent;
	status = hsa_iterate_agents(check_gpu_agent_cb, &gpu_agent);
	if (status == HSA_STATUS_INFO_BREAK) status = HSA_STATUS_SUCCESS;
	check(Find gpu agent, status);

	// Query agent name
	char name[64] = {0};
	status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, &name);
	check(Query agent name, status);
	printf("The agent name is %s\n", name);

	// Create user queue
	hsa_queue_t *queue;
	status = hsa_queue_create(gpu_agent, 64, HSA_QUEUE_TYPE_SINGLE,
				  NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
	check(Create a user queue, status);

	// Initialize execuble
	hsa_executable_t executable;
	status = hsa_executable_create_alt(HSA_PROFILE_FULL,
				       HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
				       NULL, // options
				       &executable);
	check(Create executable, status);

	// Read code object
	int fd = open("vector_add.co", O_RDONLY);
	hsa_code_object_reader_t reader;
	status = hsa_code_object_reader_create_from_file(fd, &reader);
	check(Read code object from file, status);

	// Load code object
	status = hsa_executable_load_agent_code_object(executable, gpu_agent,
			reader, NULL, NULL);
	check(Load code object, status);

	// Clean reader
	status = hsa_code_object_reader_destroy(reader);
	check(Destory reader, status);

	// Freeze executable, it could be read for symbols
	status = hsa_executable_freeze(executable, NULL);
	check(Freeze executable, status);

	// Initialize package
	uint64_t index = hsa_queue_add_write_index_screlease(queue, 1);
	//hsa_kernel_dispatch_packet_t *packet = &(((hsa_kernel_dispatch_packet_t*)queue->base_address)[index % queue->size]);
	hsa_kernel_dispatch_packet_t *packet = (hsa_kernel_dispatch_packet_t*)queue->base_address + index;

	status = init_packet(&gpu_agent, &executable, packet);
	check(Initialize packet, status);

	// Create completion signal
	status = hsa_signal_create(1, 0, NULL, &packet->completion_signal);
	check(Create completion signal, status);

	// Update package header at last
	uint16_t header = 0;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
	header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
	__atomic_store_n((uint16_t*)(&packet->header), header, __ATOMIC_RELEASE);

	// Submit package to GPU
	hsa_signal_store_screlease(queue->doorbell_signal, index);
	printf("Submit the packet to GPU\n");

	// Wait for command to finish
	hsa_signal_value_t value = hsa_signal_wait_scacquire(
			packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
			UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
	printf("signal return value %ld\n", value);

	// Check the result
	for (int i = 0; i < V_LEN; i++)
		printf("%f ", out_C[i]);
	printf("\n");

	// Clean up resources

//	free(in_A);
//	free(in_B);
//	free(out_C);
	hsa_signal_destroy(packet->completion_signal);
	hsa_queue_destroy(queue);
	hsa_shut_down();

	return 0;
}
