#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

/*
 * MPI_Sendrecv(
 *     void *sendbuf,           // send buffer
 *     int sendcount,           // number of elements in send buffer
 *     MPI_Datatype sendtype,
 *     int dest,                // rank of target process
 *     int sendtag,             // message tag for sending
 *     void *recvbuf,           // receive buffer
 *     int recvcount,           // at most the number of elements to receive
 *     MPI_Datatype recvtype,
 *     int source,              // rank of process from which to receive data
 *     int recvtag,             // message tag for receiving
 *     MPI_Comm comm,
 *     MPI_Status *status)      // info of received message
 *
 *   Send and Recv will both be done by the same process.
 */
int main(int argc, char** argv)
{
    int prev, next;
    int s_buf = -1, r_buf = -1;

    MPI_Init(NULL, NULL);

    int num_process;
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if (num_process < 3) {
        fprintf(stderr, "the number of process size should be more then 3\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    next = (my_id + 1) % num_process;
    prev = my_id - 1;
    if (prev < 0)
        prev = num_process - 1;
    s_buf = my_id;

    // my_id sends to prev
    // my_id receives from next
    MPI_Sendrecv(&s_buf, 1, MPI_INT, prev, 137,
                 &r_buf, 1, MPI_INT, next, 137,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    sleep(my_id); // show the message in order
    printf("pid: %d, rank: %d/%d, send prev(%d): 0x%X, recv next(%d): 0x%X\n",
        getpid(), my_id, num_process, prev, s_buf, next, r_buf);

    MPI_Finalize();

    return 0;
}
