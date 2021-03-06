#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#include "global.h"

#define TEST_CNT	2000
#define GROUP_NUM(arr)	(sizeof(arr) / sizeof(size_t))

static size_t group_all_sz[18] = {
	KB(4),
	KB(8),
	KB(16),
	KB(32),
	KB(64),
	KB(128),
	KB(256),
	KB(512),
	MB(1),
	MB(2),
	MB(4),
	MB(8),
	MB(16),
	MB(32),
	MB(64),
	MB(128),
	MB(256),
	MB(512),
};

static size_t group_small_sz[8] = {
	KB(4),
	KB(8),
	KB(16),
	KB(32),
	KB(64),
	KB(128),
	KB(256),
	KB(512),
};

static size_t group_big_sz[10] = {
	MB(1),
	MB(2),
	MB(4),
	MB(8),
	MB(16),
	MB(32),
	MB(64),
	MB(128),
	MB(256),
	MB(512),
};

hsa_region_t sys_region, loc_region;
hsa_agent_t gpu_agent;
hsa_agent_t cpu_agent;

enum copy_dir_type {
	COPY_D2H = 1,
	COPY_D2D,
	COPY_H2D,
};

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

static hsa_status_t get_cpu_agent_cb(hsa_agent_t agent, void *data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type == HSA_DEVICE_TYPE_CPU) {
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

static void show_result(enum copy_dir_type copy,
		struct timespec *t1,
		struct timespec *t2,
		size_t size, int count, bool cache)
{
	double start = t1->tv_sec * 1e9 + t1->tv_nsec;
	double end = t2->tv_sec * 1e9 + t2->tv_nsec;
	double rate = size / ((end - start) / count);
	char *name = NULL;

	switch (copy) {
	case COPY_H2D:
		name = "H2D";
		break;
	case COPY_D2H:
		name = "D2H";
		break;
	case COPY_D2D:
		name = "D2D";
		break;
	}

	if (test_mode == TEST_BIG) {
		printf("[%s] size: %8ld MB %10s copy rate: %10f GB/s\n",
				name,
				size >> 20,
				cache ? "cached" : "uncached",
				rate);
	} else {
		printf("[%s] size: %8ld KB %10s copy rate: %10f GB/s\n",
				name,
				size >> 10,
				cache ? "cached" : "uncached",
				rate);
	}
}

static void test_init(void)
{
	hsa_iterate_agents(get_gpu_agent_cb, &gpu_agent);
	hsa_iterate_agents(get_cpu_agent_cb, &cpu_agent);

	char name[64] = {0};
	hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, &name);
	DPT("The agent name is %s\n", name);

	hsa_agent_iterate_regions(gpu_agent,
			get_system_memory_region_cb,
			&sys_region);
	hsa_agent_iterate_regions(gpu_agent,
			get_local_memory_region_cb,
			&loc_region);
}

static void init_resource(enum copy_dir_type copy,
		hsa_region_t **src_r,
		hsa_region_t **dst_r,
		hsa_agent_t **src_a,
		hsa_agent_t **dst_a)

{
	switch (copy) {
	case COPY_H2D:
		*src_r = &sys_region;
		*dst_r = &loc_region;
		*src_a = &cpu_agent;
		*dst_a = &gpu_agent;
		break;
	case COPY_D2H:
		*src_r = &loc_region;
		*dst_r = &sys_region;
		*src_a = &gpu_agent;
		*dst_a = &cpu_agent;
		break;
	case COPY_D2D:
		*src_r = &loc_region;
		*dst_r = &loc_region;
		*src_a = &gpu_agent;
		*dst_a = &gpu_agent;
		break;
	default:
		printf("Error: Invalid copy type!\n");
	}
}

static void copy_data(enum copy_dir_type copy_type,
		size_t size,
		int count,
		bool cache,
		bool warmup)
{
	char *src = NULL;
	char *dst = NULL;
	size_t size_total;
	struct timespec t1, t2;
	hsa_region_t *src_region = NULL;
	hsa_region_t *dst_region = NULL;
	hsa_agent_t *src_agent = NULL;
	hsa_agent_t *dst_agent = NULL;

	if (cache)
		size_total = size;
	else
		size_total = size * count;

	init_resource(copy_type, &src_region, &dst_region, &src_agent, &dst_agent);

	hsa_memory_allocate(*src_region, size_total, (void **)&src);
	memset(src, 0xA, size_total);

	hsa_memory_allocate(*dst_region, size_total, (void **)&dst);
	memset(dst, 0xB, size_total);

	hsa_signal_t complete_signal;
	hsa_signal_create(1, 0, NULL, &complete_signal);

	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	for (int i = 0; i < count; ++i) {
		hsa_amd_memory_async_copy(dst, *dst_agent, src, *src_agent,
				size, 0, NULL, complete_signal);
		while (hsa_signal_wait_scacquire(complete_signal, HSA_SIGNAL_CONDITION_LT, 1,
					UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
		hsa_signal_store_screlease(complete_signal, 1);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);

	hsa_signal_destroy(complete_signal);

	hsa_memory_free(src);
	hsa_memory_free(dst);

	if (warmup)
		return;

	show_result(copy_type, &t1, &t2, size, count, cache);
}

#if 0  // for debug only
static void test_d2d(size_t size, int count, bool cache, bool warmup)
{
	hsa_region_t region;
	char *src = NULL;
	char *dst = NULL;
	struct timespec t1, t2;

	hsa_agent_iterate_regions(gpu_agent,
			get_local_memory_region_cb,
			&region);

	hsa_memory_allocate(region, size, (void **)&src);
	memset(src, 0xA, size);

	hsa_memory_allocate(region, size, (void **)&dst);
	memset(dst, 0xB, size);

	hsa_signal_t complete_signal;
	hsa_signal_create(1, 0, NULL, &complete_signal);
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	hsa_amd_memory_async_copy(dst, gpu_agent, src, gpu_agent,
			size, 0, NULL, complete_signal);
	while (hsa_signal_wait_scacquire(complete_signal, HSA_SIGNAL_CONDITION_LT, 1,
				UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
	hsa_signal_destroy(complete_signal);
	DPT("memory value [8] 0x%X  [24] 0x%X\n", dst[8], dst[24]);

	double start = t1.tv_sec * 1e6 + t1.tv_nsec / 1000;
	double end = t2.tv_sec * 1e6 + t2.tv_nsec / 1000;
	double rate = size / (end - start);
	printf("data (0x%lx) rate is %f MB/s\n", size, rate);

	hsa_memory_free(src);
	hsa_memory_free(dst);
}
#endif

static void test_copy(enum copy_dir_type copy_t, size_t *test_size, int num)
{
	int count = TEST_CNT;

	// warm up
	copy_data(copy_t, KB(4), 1, false, true);

	for (int i = 0; i < num; ++i) {
		if (test_size[i] >= MB(64))
			count = 2;
		else if (test_size[i] >= MB(1))
			count = 20;

		//copy_data(copy_t, test_size[i], count, false, false);
		copy_data(copy_t, test_size[i], count, true, false);
	}
}

int main(int argc, char **argv)
{
	get_opts(argc, argv);

	hsa_init();
	test_init();

	switch (test_mode) {
	case TEST_ALL:
		test_copy(COPY_D2D, group_all_sz, GROUP_NUM(group_all_sz));
		test_copy(COPY_H2D, group_all_sz, GROUP_NUM(group_all_sz));
		test_copy(COPY_D2H, group_all_sz, GROUP_NUM(group_all_sz));
		break;
	case TEST_SMALL:
		test_copy(COPY_D2D, group_small_sz, GROUP_NUM(group_small_sz));
		test_copy(COPY_H2D, group_small_sz, GROUP_NUM(group_small_sz));
		test_copy(COPY_D2H, group_small_sz, GROUP_NUM(group_small_sz));
		break;
	case TEST_BIG:
		test_copy(COPY_D2D, group_big_sz, GROUP_NUM(group_big_sz));
		test_copy(COPY_H2D, group_big_sz, GROUP_NUM(group_big_sz));
		test_copy(COPY_D2H, group_big_sz, GROUP_NUM(group_big_sz));
		break;
	default:
		break;
	}

	hsa_shut_down();

	return 0;
}
