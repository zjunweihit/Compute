#include <unistd.h>
#include <stdio.h>

#define check(msg, status)						\
do {									\
	if (status == HSA_STATUS_SUCCESS) {				\
		printf("%s: succeeded\n", #msg);			\
	} else {							\
		printf("%s: failed\n", #msg);				\
		exit(1);						\
	}								\
} while(0)

#define DPT(args...)							\
do {									\
	if(debug)							\
		printf(" => "args);						\
} while (0)

#define TEST_ALL        0
#define TEST_SMALL      1
#define TEST_BIG        2

#define KB(n)		((n) << 10)
#define MB(n)		((n) << 20)

extern int debug;
extern int test_mode;
extern int test_cpdma;
void get_opts(int argc, char **argv);
