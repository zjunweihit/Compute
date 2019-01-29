#include <mpi.h>
#include <stdio.h>

/*
 * MPI_Init(
 *     int* argc,
 *     char*** argv)
 *
 *   global and internal variables are set up.
 *     a communicator is created for all processes
 *     unique ranks are assigned to each process.
 *
 * MPI_Comm_size(
 *     MPI_Comm communicator,
 *     int* size)
 *
 *   return the size of communicator
 *   MPI_COMM_WORLD includes all processes.
 *
 * MPI_Comm_rank(
 *     MPI_Comm communicator,
 *     int* rank)
 *
 *   return the rank of a process in a communicator.
 *   each process of a communicator is assigned a rank inscreasing from 0.
 *
 * MPI_Get_processor_name(
 *     char* name,
 *     int* name_length)
 *
 *   get processor name, hostname
 *
 * MPI_Finalize()
 *   clean up the environment, no more MPI functions after that.
 */
int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);

    printf("Hello from %s, rank %d out of %d processes\n",
            name, world_rank, world_size);

    MPI_Finalize();

    return 0;
}
