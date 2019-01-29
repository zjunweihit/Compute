#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

/*
 * Pass the token from root to next process one by one and back to the root
 * at last, like a ring
 *
 * 0 -> 1 -> 2 -> 3 ... -> 0
 */
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

    int token = -1;

    if (world_rank != 0) {
        // Wait for previous process to send the token,
        // Root process should start the ring
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("pid: %d, rank: %d/%d, receive data: 0x%X ",
                getpid(), world_rank, world_size, token);
        token++;
        printf("update: 0x%X\n", token);
    } else {
        // Root process initialize the token
        token = 0xA000;
    }

    // Root sends the token first, then 1, 2, ...
    MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0,
             MPI_COMM_WORLD);

    // Root receives token from the previous one.
    if (world_rank == 0) {
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Root] pid: %d, rank: %d/%d, receive data: 0x%X\n",
                getpid(), world_rank, world_size, token);
    }

    MPI_Finalize();

    return 0;
}
