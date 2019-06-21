#include "global.h"

#ifdef DEBUG
int debug = 1;
#else
int debug = 0;
#endif

void get_opts(int argc, char **argv)
{
	char ch;
	while ((ch = getopt(argc, argv, "d")) != -1) {
		switch (ch) {
		case 'd':
			debug = 1;
			break;
		default:
			break;
		}
	}
}
