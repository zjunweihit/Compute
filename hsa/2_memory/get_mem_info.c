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

static hsa_status_t check_memory_pool_cb(hsa_amd_memory_pool_t pool, void *data)
{
	hsa_amd_segment_t segment;
	hsa_amd_memory_pool_get_info(
			pool,
			HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
			&segment);
	DPT("segment %d\n", segment);

	hsa_region_global_flag_t flags;
	hsa_amd_memory_pool_get_info(
			pool,
			HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
			&flags);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
		DPT("flags: kernel args\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED)
		DPT("flags: coarse grained\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)
		DPT("flags: fine grained\n");

	bool alloc = false;
	hsa_amd_memory_pool_get_info(
			pool,
			HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
			&alloc);
	DPT("runtime allocation: %s\n", alloc ? "TRUE" : "FALSE");
	if (alloc) {
		hsa_amd_memory_pool_t *p = (hsa_amd_memory_pool_t*)data;
		*p = pool;
	}

	bool host_acc = false;
	hsa_amd_memory_pool_get_info(
			pool,
			HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
			&host_acc);
	DPT("host accessible: %s\n", host_acc ? "TRUE" : "FALSE");

	return HSA_STATUS_SUCCESS;
}

static hsa_status_t check_memory_region_cb(hsa_region_t region, void* data)
{
	hsa_region_segment_t segment;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_SEGMENT,
			&segment);
	DPT("segment %d\n", segment);

	hsa_region_global_flag_t flags;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_GLOBAL_FLAGS,
			&flags);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
		DPT("flags: kernel args\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED)
		DPT("flags: coarse grained\n");
	else if(flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)
		DPT("flags: fine grained\n");

	bool alloc = false;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
			&alloc);
	DPT("runtime allocation: %s\n", alloc ? "TRUE" : "FALSE");
	if (alloc) {
		hsa_region_t *p = (hsa_region_t*)data;
		*p = region;
	}

	bool host_acc = false;
	hsa_region_get_info(
			region,
			HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
			&host_acc);
	DPT("host accessible: %s\n", host_acc ? "TRUE" : "FALSE");

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

	// Iterate memory pools
	hsa_amd_memory_pool_t mem_pool;
	status = hsa_amd_agent_iterate_memory_pools(gpu_agent,
			check_memory_pool_cb,
			&mem_pool);
	check(Iterate memory pool, status);

	// Iterate memory regions
	hsa_region_t region;
	status = hsa_agent_iterate_regions(gpu_agent,
			check_memory_region_cb,
			&region);
	check(Iterate memory region, status);

	// Clean up resources
	hsa_shut_down();

	return 0;
}
