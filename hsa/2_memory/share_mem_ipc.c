#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#include "global.h"

#define MAX_AGENT	16
#define SMALL_SIZE	(4 << 10)
#define LARGE_SIZE	(1 << 28)

struct agent_array_t
{
	hsa_agent_t agents[MAX_AGENT];
	int num;
};

static hsa_status_t get_all_agent_cb(hsa_agent_t agent, void *data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	DPT("device type %d ", device_type);

#if 0 // test for p2p
	if (device_type != HSA_DEVICE_TYPE_GPU)
		return HSA_STATUS_SUCCESS;
#endif

	struct agent_array_t *p = (struct agent_array_t*)data;
	printf("found agent 0x%lx, num %d\n", agent.handle, p->num);

	p->agents[p->num++] = agent;

	if (p->num >= MAX_AGENT)
		return HSA_STATUS_INFO_BREAK;

	return HSA_STATUS_SUCCESS;
}

static hsa_status_t get_agent(int index, hsa_agent_t *data)
{
	static struct agent_array_t agents; // it's seperate for each process

	if (agents.num == 0)
		hsa_iterate_agents(get_all_agent_cb, &agents);

	if (index > agents.num) {
		printf("Error: invalid index(%d) over max agent num(%d)\n",
				index, agents.num);
		return HSA_STATUS_ERROR_INVALID_ARGUMENT;
	}
	*data = agents.agents[index];

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

void export_memory(int fd)
{
	hsa_status_t status;
	hsa_agent_t agent;
	hsa_region_t region;
	int *ptr = NULL;
	size_t size = LARGE_SIZE;
	hsa_amd_ipc_memory_t handle = {0};

	status = hsa_init();
	check([export] initialize HSA, status);

	status = get_agent(1, &agent);
	check([export] get agent, status);

	// Get local memory regions
	status = hsa_agent_iterate_regions(agent,
			get_local_memory_region_cb,
			&region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check([export] get local memory region, status);

	status = hsa_memory_allocate(region, size, (void **)&ptr);
	memset(ptr, 0xA, size);
	check([export] allocate local memory, status);

	status = hsa_amd_ipc_memory_create(ptr, size, &handle);
	check([export] export ipc memory, status);

	write(fd, &handle, sizeof(handle));

	sleep(3); // make sure exported memory is not used any more

	status = hsa_memory_free(ptr);
	check([export] free export memory, status);

	status = hsa_shut_down();
	check([export] shutdown HSA, status);
}

void import_memory(int fd)
{
	hsa_status_t status;
	hsa_agent_t agent;
	hsa_region_t region;
	size_t size = LARGE_SIZE;
	hsa_amd_ipc_memory_t handle = {0};
	char *src = NULL; // from export memory
	char *dst = NULL; // to local memory

	status = hsa_init();
	check([import] initialize HSA, status);

	read(fd, &handle, sizeof(handle));
	printf("handle[0] %u\n", handle.handle[0]);

	status = get_agent(1, &agent); // we can use another gpu 2 for p2p
	check([import] get agent, status);

	// import memory from handle
	status = hsa_amd_ipc_memory_attach(&handle, size, 1, &agent, (void**)&src);
	check([import] Import ipc memory, status);

	// Get local memory regions
	status = hsa_agent_iterate_regions(agent,
			get_local_memory_region_cb,
			&region);
	status = status == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_INFO_BREAK;
	check([import] get local memory region, status);

	status = hsa_memory_allocate(region, size, (void **)&dst);
	memset(dst, 0xB, size);
	check([import] allocate local memory, status);

	hsa_signal_t complete_signal;
	status = hsa_signal_create(1, 0, NULL, &complete_signal);
	check([import] create complete signal, status);
	status = hsa_amd_memory_async_copy(dst, agent, src, agent,
					   16, 0, NULL, complete_signal);
	while (hsa_signal_wait_scacquire(complete_signal,
				HSA_SIGNAL_CONDITION_EQ, 0,
				UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
	hsa_signal_destroy(complete_signal);
	check([import] hsa memory async copy, status);
	DPT("memory value [8] 0x%X  [24] 0x%X\n", dst[8], dst[24]);

	status = hsa_memory_free(dst);
	check([import] free local memory, status);

	status = hsa_shut_down();
	check([import] shutdown HSA, status);
}

int main(int argc, char **argv)
{
	pid_t pid;
	int fd[2];

	get_opts(argc, argv);

	assert(!pipe(fd) && "create pipe");

	pid = fork();
	if (pid) // parent
		export_memory(fd[1]);
	else
		import_memory(fd[0]);

	waitpid(pid, NULL, 0);

	return 0;
}
