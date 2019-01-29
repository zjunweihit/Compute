#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

/*
 * MPI_Send(
 *     void* data,              // data buffer
 *     int count,               // count of elements in the buffer to send
 *     MPI_Datatype datatype,   // the type of data, e.g. MPI_INT, MPI_FLOAT
 *     int destination,         // rank of target process
 *     int tag,                 // the tag of message
 *     MPI_Comm communicator)   // specify the communicator
 *
 * MPI_Recv(
 *     void* data,
 *     int count,               // receive at most the count of elements
 *     MPI_Datatype datatype,
 *     int source,
 *     int tag,
 *     MPI_Comm communicator,
 *     MPI_Status* status)      // info of received message
 */
int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 2) {
        fprintf(stderr, "World size should be 2 for this test\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int number = -1;
    if (world_rank == 0) {
        number = 0xABCD;
        MPI_Send(
          /* data         = */ &number,
          /* count        = */ 1,
          /* datatype     = */ MPI_INT,
          /* destination  = */ 1,
          /* tag          = */ 5,
          /* communicator = */ MPI_COMM_WORLD);
        printf("pid: %d, rank: %d/%d, send data: 0x%X\n",
                getpid(), world_rank, world_size, number);
    } else {
        MPI_Recv(
          /* data         = */ &number,
          /* count        = */ 1,
          /* datatype     = */ MPI_INT,
          /* source       = */ 0,
          /* tag          = */ 5,
          /* communicator = */ MPI_COMM_WORLD,
          /* status       = */ MPI_STATUS_IGNORE);

        printf("pid: %d, rank: %d/%d, receive data: 0x%X\n",
                getpid(), world_rank, world_size, number);
    }

    MPI_Finalize();

    return 0;
}
