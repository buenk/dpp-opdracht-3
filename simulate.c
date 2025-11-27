/*
 * simulate.c
 *
 * Contains the implementation of the main simulation logic,
 * including parallelization of computations for the wave equation
 * using MPI for distributed computing.
 *
 * Authors: Ezra Buenk, Anais Marelis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "simulate.h"


/*
 * Initializes the local domain for this MPI process. Calculates the chunk
 * this process is responsible for and allocates local arrays with halo cells.
 *
 * rank: MPI rank of this process
 * size: total number of MPI processes
 * i_max: total number of data points on a single wave
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * local_n: pointer to store the number of local points (output)
 * start_index: pointer to store the global starting index (output)
 * local_old: pointer to store allocated local old array (output)
 * local_current: pointer to store allocated local current array (output)
 * local_next: pointer to store allocated local next array (output)
 */
static void init_local_domain(int rank, int size, int i_max,
		double *old_array, double *current_array,
		int *local_n, int *start_index,
		double **local_old, double **local_current, double **local_next)
{
	int remainder = i_max % size;
	*local_n = i_max / size;

	// Handle uneven sizes of domains.
	if (rank < remainder) {
		(*local_n)++;
		*start_index = rank * (*local_n);
	} else {
		*start_index = remainder * ((*local_n) + 1) + (rank - remainder) * (*local_n);
	}

	// Allocate local arrays with halo cells (+2 for left and right ghost).
	int local_size = (*local_n) + 2;
	*local_old = malloc(local_size * sizeof(double));
	*local_current = malloc(local_size * sizeof(double));
	*local_next = malloc(local_size * sizeof(double));

	// Index 0 is left halo, index local_n + 1 is right halo.
	// Everything in between is actual data.
	for (int i = 0; i < *local_n; i++) {
		(*local_old)[i + 1] = old_array[*start_index + i];
		(*local_current)[i + 1] = current_array[*start_index + i];
	}
	(*local_next)[0] = 0.0;
	(*local_next)[local_size - 1] = 0.0;
}

/*
 * Exchanges halo cells with neighboring processes using blocking MPI calls.
 *
 * local_current: local array containing current wave data with halo cells
 * local_n: number of local data points (excluding halos)
 * left_neighbour: MPI rank of left neighbour (or MPI_PROC_NULL)
 * right_neighbour: MPI rank of right neighbour (or MPI_PROC_NULL)
 */
static void exchange_halos(double *local_current, int local_n,
		int left_neighbour, int right_neighbour)
{
	// Send leftmost real cell to left neighbour, receive right halo from right neighbour.
	MPI_Sendrecv(&local_current[1], 1, MPI_DOUBLE, left_neighbour, 0,
	             &local_current[local_n + 1], 1, MPI_DOUBLE, right_neighbour, 0,
	             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Send rightmost real cell to right neighbour, receive left halo from left neighbour.
	MPI_Sendrecv(&local_current[local_n], 1, MPI_DOUBLE, right_neighbour, 1,
	             &local_current[0], 1, MPI_DOUBLE, left_neighbour, 1,
	             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/*
 * Computes one time step of the wave equation for the local domain.
 *
 * local_old: local array containing wave data for t-1
 * local_current: local array containing wave data for t
 * local_next: local array to store wave data for t+1 (output)
 * local_n: number of local data points (excluding halos)
 * start_index: global starting index for this process's chunk
 * i_max: total number of data points on a single wave
 */
static void compute_wave_step(double *local_old, double *local_current,
		double *local_next, int local_n, int start_index, int i_max)
{
    const double c = 0.15;

	for (int i = 1; i <= local_n; i++) {
		int global_i = start_index + (i - 1);
		if (global_i == 0 || global_i == i_max - 1) {
			local_next[i] = 0.0;
		} else {
			local_next[i] = 2.0 * local_current[i] - local_old[i]
				+ c * (local_current[i - 1] - 2.0 * local_current[i]
					+ local_current[i + 1]);
		}
	}
}

/*
 * Execute the entire wave equation simulation using MPI. Divides the wave
 * points among processes using domain decomposition, exchanging halo cells
 * between neighbours after each timestep.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 *
 * Returns:
 * A pointer to the final wave state (gathered on rank 0)
 */
double *simulate(const int i_max, const int t_max, double *old_array,
		double *current_array, double *next_array)
{
	MPI_Init(NULL, NULL);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Start timing after MPI init
	double start_time = MPI_Wtime();

	// Initialize local domain.
	int local_n, start_index;
	double *local_old, *local_current, *local_next;
	init_local_domain(rank, size, i_max, old_array, current_array,
			&local_n, &start_index, &local_old, &local_current, &local_next);

	int left_neighbour = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
	int right_neighbour = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

	for (int t = 0; t < t_max; t++) {
		exchange_halos(local_current, local_n, left_neighbour, right_neighbour);
		compute_wave_step(local_old, local_current, local_next,
				local_n, start_index, i_max);

		double *temp = local_old;
		local_old = local_current;
		local_current = local_next;
		local_next = temp;
	}

	// Gather results back to rank 0.
	int remainder = i_max % size;
	int *recvcounts = NULL;
	int *displs = NULL;

	if (rank == 0) {
		recvcounts = malloc(size * sizeof(int));
		displs = malloc(size * sizeof(int));

		int offset = 0;
		for (int p = 0; p < size; p++) {
			int p_local_n = i_max / size + (p < remainder ? 1 : 0);
			recvcounts[p] = p_local_n;
			displs[p] = offset;
			offset += p_local_n;
		}
	}

	MPI_Gatherv(&local_current[1], local_n, MPI_DOUBLE,
	            current_array, recvcounts, displs, MPI_DOUBLE,
	            0, MPI_COMM_WORLD);

	// End timing before MPI finalize.
	double end_time = MPI_Wtime();
	double compute_time = end_time - start_time;

	// Print actual computation time (excluding MPI_Init/Finalize overhead)
	if (rank == 0) {
		printf("Compute time: %g seconds\n", compute_time);
	}

	free(local_old);
	free(local_current);
	free(local_next);
	if (rank == 0) {
		free(recvcounts);
		free(displs);
	}

	MPI_Finalize();

	return current_array;
}
