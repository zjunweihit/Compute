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

extern int debug;
void get_opts(int argc, char **argv);
