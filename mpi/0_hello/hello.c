#include <mpi.h>
#include <stdio.h>

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
