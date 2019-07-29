#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#include "global.h"

static hsa_status_t get_gpu_agent_cb(hsa_agent_t agent, void *data)
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

static hsa_status_t get_local_memory_region_cb(hsa_region_t region, void* data)
{
	hsa_region_segment_t segment;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_SEGMENT,
			&segment);
	if (segment != HSA_REGION_SEGMENT_GLOBAL)
		goto out;

	bool host_acc = false;
	hsa_region_get_info(
			region,
			HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
			&host_acc);
	if (host_acc)
		goto out;

	hsa_region_global_flag_t flags;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_GLOBAL_FLAGS,
			&flags);

	DPT("segment %d ", segment);
	DPT("host accessible: %s ", host_acc ? "TRUE" : "FALSE");
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
		DPT("flags: kernel args\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED)
		DPT("flags: coarse grained\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)
		DPT("flags: fine grained\n");

	hsa_region_t *p = (hsa_region_t*)data;
	*p = region;

	return HSA_STATUS_INFO_BREAK;
out:
	return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
	get_opts(argc, argv);

	hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

	// Find gpu agent
	hsa_agent_t gpu_agent;
	status = hsa_iterate_agents(get_gpu_agent_cb, &gpu_agent);
	if (status == HSA_STATUS_INFO_BREAK) status = HSA_STATUS_SUCCESS;
	check(Find gpu agent, status);

	// Query agent name
	char name[64] = {0};
	status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, &name);
	check(Query agent name, status);
	DPT("The agent name is %s\n", name);

	// Create user memory
	char *user_ptr;
	int size = MB(8);
	assert(!posix_memalign((void**)&user_ptr, 4096, size));
	memset(user_ptr, 0xA, size);
	DPT("user memory value is 0x%X\n", user_ptr[16]);

	char *lock_ptr;
	status = hsa_amd_memory_lock(user_ptr,
			size,
			&gpu_agent,	// pointer to agent array
			1,		// number of agent
			(void**)&lock_ptr);	// locked ptr for agents
	check(lock user memory to gpu, status);

	// Get local memory regions
	hsa_region_t region;
	status = hsa_agent_iterate_regions(gpu_agent,
			get_local_memory_region_cb,
			&region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get system local region, status);

	// Allocate local memory
	char *dev_ptr;
	status = hsa_memory_allocate(region, size, (void **)&dev_ptr);
	check(Allocate local memory, status);
	memset(dev_ptr, 0xB, size);
	DPT("memory value is 0x%X\n", dev_ptr[16]);

	status = hsa_memory_copy(lock_ptr,	// dst
				dev_ptr,	// src
				size);
	check(hsa memory copy, status);
	DPT("memory value [16] 0x%X  [32] 0x%X\n", user_ptr[16], user_ptr[32]);

	free(user_ptr);
	status = hsa_memory_free(dev_ptr);
	check(Free small local memory, status);

	hsa_shut_down();

	return 0;
}

