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

hsa_agent_t gpu_agent;

static void test_init(void)
{
	hsa_iterate_agents(get_gpu_agent_cb, &gpu_agent);

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

static void get_region(enum copy_dir_type copy, hsa_region_t **src, hsa_region_t **dst)
{
	switch (copy) {
	case COPY_H2D:
		*src = &sys_region;
		*dst = &loc_region;
		break;
	case COPY_D2H:
		*src = &loc_region;
		*dst = &sys_region;
		break;
	case COPY_D2D:
		*src = &loc_region;
		*dst = &loc_region;
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
	hsa_region_t *src_region, *dst_region;

	if (cache)
		size_total = size;
	else
		size_total = size * count;

	get_region(copy_type, &src_region, &dst_region);

	hsa_memory_allocate(*src_region, size_total, (void **)&src);
	memset(src, 0xA, size_total);
	hsa_memory_allocate(*dst_region, size_total, (void **)&dst);
	memset(dst, 0xB, size_total);

	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	for (int i = 0; i < count; ++i) {
		if (cache)
			hsa_memory_copy(dst, src, size);
		else
			hsa_memory_copy(dst + size * i, src + size * i, size);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);

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
	size_t size_total;
	struct timespec t1, t2;

	if (cache)
		size_total = size;
	else
		size_total= size * count;


	hsa_agent_iterate_regions(gpu_agent,
			get_local_memory_region_cb,
			&region);

	hsa_memory_allocate(region, size_total, (void **)&src);
	memset(src, 0xA, size_total);
	hsa_memory_allocate(region, size_total, (void **)&dst);
	memset(dst, 0xB, size_total);

	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	for (int i = 0; i < count; ++i) {
		if (cache)
			hsa_memory_copy(dst, src, size);
		else
			hsa_memory_copy(dst + size * i, src + size * i, size);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);

	hsa_memory_free(src);
	hsa_memory_free(dst);

	if (warmup)
		return;

	show_result(COPY_D2D, &t1, &t2, size, count, cache);
}
#endif

static void test_copy(enum copy_dir_type copy_t, size_t *test_size, int num)
{
	int count = TEST_CNT;
	size_t size;

	// warm up
	copy_data(copy_t, KB(4), 1, false, true);

	for (int i = 0; i < num; ++i) {
		size = test_size[i];
		if (size >= MB(64)) {
			count = 2;
			if (test_cpdma)
				size = MB(64) - 1;
		} else if (size >= MB(1)) {
			count = 20;
		}

		//copy_data(copy_t, test_size[i], count, false, false);
		copy_data(copy_t, size, count, true, false);
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
