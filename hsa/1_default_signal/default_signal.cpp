#include <iostream>
#include <thread>

#include <hsa.h>
#include <hsa_ext_amd.h>
#include <assert.h>

hsa_signal_t signal;

void thread1()
{
	printf("T1: waiting for signal == 1\n");
	hsa_signal_wait_scacquire(
			signal, HSA_SIGNAL_CONDITION_EQ, 1,
			UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
	printf("T1: signal == 1, T1 runs\n");

	printf("T1: update signal = 2\n");
	hsa_signal_store_screlease(signal, 2);

	printf("T1: waiting for signal == 3\n");
	hsa_signal_wait_scacquire(
			signal, HSA_SIGNAL_CONDITION_EQ, 3,
			UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
	printf("T1: signal == 3, T1 runs\n");
	printf("T1: done\n");
}

void thread2()
{
	printf("T2: update signal = 1\n");
	hsa_signal_store_screlease(signal, 1);

	printf("T2: waiting for signal update signal = 2\n");
	hsa_signal_wait_scacquire(
			signal, HSA_SIGNAL_CONDITION_EQ, 2,
			UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
	printf("T2: signal == 2, T2 runs\n");

	printf("T2: update signal 3\n");
	hsa_signal_store_screlease(signal, 3);
	printf("T2: done\n");
}

int main()
{
	int ret = 0;
	assert(hsa_init() == HSA_STATUS_SUCCESS);
	ret = hsa_amd_signal_create(0, 0, NULL, HSA_AMD_SIGNAL_IPC, &signal);
	if (ret != HSA_STATUS_SUCCESS)
		fprintf(stderr, "failed to create signal");

	printf("main: signal = 0\n");

	std::thread t1(thread1);
	std::thread t2(thread2);

	t1.join();
	t2.join();
	std::cout << "done" << std::endl;

	return 0;
}
