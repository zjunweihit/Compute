#include "global.h"
#include <stdlib.h>

#ifdef DEBUG
int debug = 1;
#else
int debug = 0;
#endif

static void print_help(void)
{
	printf("-d: show debug info\n");
	printf("-h: show this help\n");
}

void get_opts(int argc, char **argv)
{
	char ch;
	while ((ch = getopt(argc, argv, "dh")) != -1) {
		switch (ch) {
		case 'd':
			debug = 1;
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
