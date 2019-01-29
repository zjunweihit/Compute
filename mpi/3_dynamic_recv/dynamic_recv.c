#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
 * MPI_Status
 *   .MPI_SOURCE: the rank of sender
 *   .MPI_TAG: the tag of the message
 *
 * MPI_Get_count(
 *     MPI_Status* status,
 *     MPI_Datatype datatype,
 *     int* count)
 *
 *   The length of the message.
 *   User passes status, the datatype message and count is returned.
 *   Because MPI_Recv can get MPI_ANY_SOURCE and MPI_ANY_TAG, status is the only
 *   way to get the exact sender.
 *
 * MPI_Probe(
 *     int source,
 *     int tag,
 *     MPI_Comm comm,
 *     MPI_Status* status)
 *
 *   Query the message info before MPI_Recv.
 *   It's similar to MPI_Recv but receive the data actually, blocking for
 *   a message with mathched tag.
 */

#define MAX_NUM 100

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 3) {
        fprintf(stderr, "World size should be 3 for this test\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int data_array[MAX_NUM] = {0xa};
    int data_count = -1;

    if (world_rank == 0) {

        srand(time(NULL));
        data_count = ((float)rand() / RAND_MAX) * MAX_NUM;

        MPI_Send(&data_array, data_count, MPI_INT, 1, 3,
                 MPI_COMM_WORLD);

        data_count = ((float)rand() / RAND_MAX) * MAX_NUM;
        MPI_Send(&data_array, data_count, MPI_INT, 2, 5,
                 MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        // get status after receiving data from 0
        MPI_Status status;
        MPI_Recv(&data_array, MAX_NUM, MPI_INT, 0, 3,
                 MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &data_count);
        printf("pid %d rank 1 receives %d data from rank %d, tag %d\n",
                getpid(), data_count, status.MPI_SOURCE, status.MPI_TAG);
    } else if (world_rank == 2) {
        // probe status before after receiving data from 0
        MPI_Status status;
        MPI_Probe(0, 5, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &data_count);

        printf("pid %d rank 2 will receive %d data from rank %d, tag %d\n",
                getpid(), data_count, status.MPI_SOURCE, status.MPI_TAG);

        int *buf = (int*)malloc(data_count * sizeof(int));
        MPI_Recv(&data_array, MAX_NUM, MPI_INT, 0, 5,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        free(buf);
    }

    MPI_Finalize();

    return 0;
}
