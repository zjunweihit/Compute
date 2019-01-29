#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
 * MPI_Barrier(MPI_Comm communicator)
 *   No process in a communicator could pass the barrier until all of them call
 *   the function.
 *   It could be used to synchronize each portion from parallel code.
 *
 *   ** Every collective call is synchronized.**
 *
 * MPI_Bcast(
 *     void* data,              // send to all processes
 *     int count,               // the number of data in the message
 *     MPI_Datatype datatype,   // element type
 *     int root,                // who will do the broadcast
 *     MPI_Comm communicator)   // sepecify the communicator
 *
 *   root and others could call the same function.
 *   root: send data
 *   otheres: receive the data
 */

#define MAX_NUM 10

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        fprintf(stderr, "World size should be more than 2\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int data_array[MAX_NUM] = {0xa};

    MPI_Bcast(&data_array, MAX_NUM, MPI_INT, 0, MPI_COMM_WORLD);

    printf("rank %d/%d received data from 0\n", world_rank, world_size);

    MPI_Finalize();

    return 0;
}
