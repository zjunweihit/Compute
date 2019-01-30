#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
 * MPI_Allgather(
 *     void* send_data,             // data array of sub buffer
 *     int send_count,              // number of data from sub buffer
 *     MPI_Datatype send_datatype,  // data type of send data
 *     void* recv_data,             // recv data array to gather data from all
 *     int recv_count,              // number of data from **each** process's sub buffer
 *     MPI_Datatype recv_datatype,  // data type of recv data
 *     MPI_Comm communicator)       // specify the communicator
 *
 *   There is no root process, since every one gathers data from all.
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

    int data_gather[MAX_NUM] = {0};
    MPI_Allgather(sub_buf, count_per_process, MPI_INT,
               &data_gather, count_per_process, MPI_INT,
               MPI_COMM_WORLD);

    // show data of all processes one by one
    sleep(world_rank);
    printf("process %d gathers data: ", world_rank);
    for (i = 0; i < MAX_NUM; i++)
        printf("%d ", data_gather[i]);
    printf("\n");

    free(sub_buf);
    MPI_Finalize();

    return 0;
}
