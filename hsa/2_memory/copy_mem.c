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

static hsa_status_t get_system_memory_region_cb(hsa_region_t region, void* data)
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
	if (!host_acc)
		goto out;

	hsa_region_global_flag_t flags;
	hsa_region_get_info(
			region,
			HSA_REGION_INFO_GLOBAL_FLAGS,
			&flags);
	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
		goto out;

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

#define SMALL_SIZE (4 << 10)
#define LARGE_SIZE (4 << 20)

static hsa_status_t test_h2h(hsa_agent_t agent, size_t size)
{
	hsa_status_t status;
	hsa_region_t region;
	char *src = NULL;
	char *dst = NULL;

	// Get system memory regions
	status = hsa_agent_iterate_regions(agent,
			get_system_memory_region_cb,
			&region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get system memory region, status);

	// Allocate system memory
	status = hsa_memory_allocate(region, size, (void **)&src);
	check(Allocate system memory src, status);
	memset(src, 0xA, size);
	DPT("memory value is 0x%X\n", src[16]);

	status = hsa_memory_allocate(region, size, (void **)&dst);
	check(Allocate system memory dst, status);
	memset(dst, 0xB, size);
	DPT("memory value is 0x%X\n", dst[16]);

	memcpy(dst, src, 10);
	DPT("memory value [8] 0x%X [16] 0x%X\n", dst[8], dst[16]);

	status = hsa_memory_copy(dst, src, 20);
	check(hsa memory copy, status);
	DPT("memory value [16] 0x%X  [32] 0x%X\n", dst[16], dst[32]);

	status = hsa_memory_free(src);
	check(Free system memory src, status);
	status = hsa_memory_free(dst);
	check(Free system memory dst, status);

	return status;
}

static hsa_status_t test_h2d(hsa_agent_t agent, size_t size)
{
	hsa_status_t status;
	hsa_region_t sys_region, loc_region;
	char *src = NULL;
	char *dst = NULL;

	// Get system memory regions
	status = hsa_agent_iterate_regions(agent,
			get_system_memory_region_cb,
			&sys_region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get system memory region, status);

	// Get local memory regions
	status = hsa_agent_iterate_regions(agent,
			get_local_memory_region_cb,
			&loc_region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get local memory region, status);

	status = hsa_memory_allocate(sys_region, size, (void **)&src);
	check(Allocate system memory src, status);
	memset(src, 0xA, size);
	DPT("memory value is 0x%X\n", src[16]);

	status = hsa_memory_allocate(loc_region, size, (void **)&dst);
	memset(dst, 0xB, size);
	check(Allocate local memory dst, status);

	status = hsa_memory_copy(dst, src, 16);
	check(hsa memory copy, status);
	DPT("memory value [8] 0x%X  [24] 0x%X\n", dst[8], dst[24]);

	status = hsa_memory_free(src);
	check(Free system memory src, status);
	status = hsa_memory_free(dst);
	check(Free local memory dst, status);

	return status;
}

static hsa_status_t test_d2h(hsa_agent_t agent, size_t size)
{
	hsa_status_t status;
	hsa_region_t sys_region, loc_region;
	char *src = NULL;
	char *dst = NULL;

	// Get system memory regions
	status = hsa_agent_iterate_regions(agent,
			get_system_memory_region_cb,
			&sys_region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get system memory region, status);

	// Get local memory regions
	status = hsa_agent_iterate_regions(agent,
			get_local_memory_region_cb,
			&loc_region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get local memory region, status);

	status = hsa_memory_allocate(loc_region, size, (void **)&src);
	check(Allocate local memory src, status);
	memset(src, 0xA, size);
	DPT("memory value is 0x%X\n", src[16]);

	status = hsa_memory_allocate(sys_region, size, (void **)&dst);
	memset(dst, 0xB, size);
	check(Allocate system memory dst, status);

	status = hsa_memory_copy(dst, src, 16);
	check(hsa memory copy, status);
	DPT("memory value [8] 0x%X  [24] 0x%X\n", dst[8], dst[24]);

	status = hsa_memory_free(src);
	check(Free system memory src, status);
	status = hsa_memory_free(dst);
	check(Free local memory dst, status);

	return status;
}

static hsa_status_t test_d2d(hsa_agent_t agent, size_t size)
{
	hsa_status_t status;
	hsa_region_t region;
	char *src = NULL;
	char *dst = NULL;

	// Get local memory regions
	status = hsa_agent_iterate_regions(agent,
			get_local_memory_region_cb,
			&region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check(Get local memory region, status);

	status = hsa_memory_allocate(region, size, (void **)&src);
	check(Allocate local memory src, status);
	memset(src, 0xA, size);
	DPT("memory value is 0x%X\n", src[16]);

	status = hsa_memory_allocate(region, size, (void **)&dst);
	memset(dst, 0xB, size);
	check(Allocate local memory dst, status);

	status = hsa_memory_copy(dst, src, 16);
	check(hsa memory copy, status);
	DPT("memory value [8] 0x%X  [24] 0x%X\n", dst[8], dst[24]);

	status = hsa_memory_free(src);
	check(Free local memory src, status);
	status = hsa_memory_free(dst);
	check(Free local memory dst, status);

	return status;
}

int main(int argc, char **argv)
{
	get_opts(argc, argv);

	hsa_status_t status;

	status = hsa_init();
	check(Initialize HSA, status);

	hsa_agent_t gpu_agent;
	status = hsa_iterate_agents(get_gpu_agent_cb, &gpu_agent);
	if (status == HSA_STATUS_INFO_BREAK) status = HSA_STATUS_SUCCESS;
	check(Find gpu agent, status);

	char name[64] = {0};
	status = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, &name);
	check(Query agent name, status);
	DPT("The agent name is %s\n", name);

	status = test_h2h(gpu_agent, SMALL_SIZE);
	check(=== test h2h ===, status);

	status = test_h2d(gpu_agent, LARGE_SIZE);
	check(=== test h2d ===, status);

	status = test_d2h(gpu_agent, LARGE_SIZE);
	check(=== test d2h ===, status);

	status = test_d2d(gpu_agent, LARGE_SIZE);
	check(=== test d2d ===, status);

	status = hsa_shut_down();
	check(Shutdown HSA, status);

	return 0;
}
