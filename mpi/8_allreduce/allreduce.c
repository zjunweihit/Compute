#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
 * MPI_Allreduce(
 *     void* send_data,             // data array that each process wants to reduce
 *     void* recv_data,             // data array only for the root process
 *     int count,                   // number of data in recv_data buffer
 *     MPI_Datatype datatype,       // datatype of elements in send_data array
 *     MPI_Op op,                   // operation applied to the data
 *     MPI_Comm communicator)
 *
 * MPI_Op type:
 *
 *   MPI_MAX    - Returns the maximum element.
 *   MPI_MIN    - Returns the minimum element.
 *   MPI_SUM    - Sums the elements.
 *   MPI_PROD   - Multiplies all elements.
 *   MPI_LAND   - Performs a logical and across the elements.
 *   MPI_LOR    - Performs a logical or across the elements.
 *   MPI_BAND   - Performs a bitwise and across the bits of the elements.
 *   MPI_BOR    - Performs a bitwise or across the bits of the elements.
 *   MPI_MAXLOC - Returns the maximum value and the rank of the process that owns it.
 *   MPI_MINLOC - Returns the minimum value and the rank of the process that owns it.
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

    int data_array[MAX_NUM] = {0};
    int count_per_process = 2;
    int i = -1;

    // root: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    if (world_rank == 0) {
        for (i = 0; i < MAX_NUM; i++)
            data_array[i] = i;
    }

    int* sub_buf = (int*)malloc(count_per_process * sizeof(int));
    MPI_Scatter(&data_array, count_per_process, MPI_INT,
                sub_buf, count_per_process, MPI_INT,
                0, MPI_COMM_WORLD);

    printf("rank %d/%d received data: ", world_rank, world_size);
    for (i = 0; i < count_per_process; i++)
        printf("%d ", sub_buf[i]);
    printf("\n");

    int *data_reduce = (int*)malloc(count_per_process * sizeof(int));
    MPI_Allreduce(sub_buf, data_reduce, count_per_process, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    sleep(world_rank);
    printf("process %d reduces(MPI_SUM) data: ", world_rank);
    for (i = 0; i < count_per_process; i++)
        printf("%d ", data_reduce[i]);
    printf("\n");

    free(sub_buf);
    free(data_reduce);
    MPI_Finalize();

    return 0;
}
