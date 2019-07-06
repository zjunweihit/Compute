#include "global.h"
#include <stdlib.h>

#ifdef DEBUG
int debug = 1;
#else
int debug = 0;
#endif

int test_mode = 0;

static void print_help(void)
{
	printf("-d: show debug info\n");
	printf("-m: set test mode for perf test\n");
	printf("    0: test all size group\n");
	printf("    1: test small size group\n");
	printf("    2: test big size group\n");
	printf("-h: show this help\n");
}

void get_opts(int argc, char **argv)
{
	char ch;
	while ((ch = getopt(argc, argv, "dm:h")) != -1) {
		switch (ch) {
		case 'd':
			debug = 1;
			break;
		case 'm':
			test_mode = atoi(optarg);
			if (test_mode > 2) {
				print_help();
				exit(0);
			}
			break;
		case 'h':
			print_help();
			exit(0);
			break;
		default:
			break;
		}
	}
}
