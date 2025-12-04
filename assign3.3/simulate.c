/*
 * simulate.c
 *
 * A MPI program that implements MYMPI_Bcast using point-to-point
 * communication on a logical 1-D ring toposolgy
 *
 * Authors: Ezra Buenk, Anais Marelis.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


/*
 * Broadcast a message from the root process to every other process
 * implemented using only point-to-point communication.
 */
int MYMPI_Bcast(
	void *buffer,
	int count,
	MPI_Datatype datatype,
	int root,
	MPI_Comm communicator
) {
	int rank, size;
	MPI_Comm_rank(communicator, &rank);
	MPI_Comm_size(communicator, &size);

	// Only one process, thus trivial
	if (size <= 1) return MPI_SUCCESS;

	int left = (rank - 1 + size) % size;
	int right = (rank + 1) % size;
	MPI_Status status;
	// relative position from the root
	int rel = (rank - root + size) % size;

	// process is the root, sends message in both ring directions.
	if (rank == root) {
		if (left == right) {
			MPI_Send(buffer, count, datatype, left, 0, communicator);
		} else {
			MPI_Send(buffer, count, datatype, right, 0, communicator);
			MPI_Send(buffer, count, datatype, left, 0, communicator);
		}
		return MPI_SUCCESS;
	} else { // non-root process, check from where the message comes
			// and broadcast to the chosen neighbour.
		int source;
		if (rel <= size / 2) {
			source = left;
		} else {
			source = right;
		}
		MPI_Recv(buffer, count, datatype, source, 0, communicator, &status);

		if (rel < size - rel) {
			int dest;
			if (source == left) {
				dest = right;
			} else {
				dest = left;
			}
			MPI_Send(buffer, count, datatype, dest, 0, communicator);
		}
		return MPI_SUCCESS;
	}
}

int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int root = 0;
	int count = 8;

	int *buff = (int*) malloc(sizeof(int) * count);
	if (!buff) {
		fprintf(stderr, "malloc failed\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (rank == root) {
		for (int i = 0; i < count; ++i) buff[i] = 1000 + i;
	} else {
		for (int i = 0; i < count; ++i) buff[i] = -1;
	}

	MYMPI_Bcast(buff, count, MPI_INT, root, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	for (int r = 0; r < size; ++r) {
		if (r == rank) {
			printf("rank %2d: buff =", rank);
			for (int i = 0; i < count; ++i) printf(" %d", buff[i]);
			printf("\n");
			fflush(stdout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	free(buff);
	MPI_Finalize();
	return 0;
}
