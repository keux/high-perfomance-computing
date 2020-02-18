#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>


void step(const std::vector<double>& Ui, std::vector<double>& Ui1, int size, double rl, double rr, double dx,
        double dt, int r, int pc)
{
    double C = dt / (dx*dx);


    for (int i = 0; i < size; ++i)
    {
        if (i == 0)
        {
            if (r == 0)
                Ui1[i] = 0;
            else
                Ui1[i] = Ui[i] + C * (rl - 2 * Ui[i] + Ui[i + 1]);
        } else if (i == size - 1) {
            if (r == pc - 1)
                Ui1[i] = 0;
            else
                Ui1[i] = Ui[i] + C * (Ui[i - 1] - 2 * Ui[i] + rr);

        } else {
            Ui1[i] = Ui[i] + C * (Ui[i - 1] - 2 * Ui[i] + Ui[i + 1]);
        }
    }
}

void printTimeRank(int threadsNum, std::vector<double>& Ui,  int size,  int rank)
{
    char name[10];
    sprintf(name, "%i_result_rank%i", threadsNum, rank);
    FILE* file = fopen(name, "a");
    for (int i = 0; i < size; i += 5)
    {
        if(i != size - 1)
            fprintf(file, "%lf\t", Ui[i]);
        else
            fprintf(file, "%lf\t\n", Ui[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}
void solveHeat(std::vector<double>& U, double a, double b, double Nx, int rank, int sizeProc, double dx, double dt, int NTimeSteps)
{

    int Nr = Nx / sizeProc;
    int start = Nr * rank;
    if (rank == sizeProc - 1)
        Nr = Nx - start;
    int end = start + Nr;
    int size = Nr;

    std::vector<double> Ui(size), Ui1(size);
    for (int i = start; i < end; ++i)
    {
        Ui[i - start] = U[start*0 + i];
        Ui1[i - start] = U[start*0 + i];
    }

    auto upTo = NTimeSteps * 10;
    for (int s = 0; s < upTo; ++s)
    {
        double sl = Ui[0], sr = Ui[size - 1];
        double rl = Ui[0], rr = Ui[size - 1];

        MPI_Status status;

        if (rank < sizeProc - 1)
            MPI_Send((&sr), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
        if (rank > 0)
            MPI_Recv((&rl), 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);

        if (rank > 0)
            MPI_Send((&sl), 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        if (rank < sizeProc - 1)
            MPI_Recv((&rr), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);


        step(Ui, Ui1, size, rl, rr, dx, dt, rank, sizeProc);
        for (int i = start; i < end; ++i)
        {
            U[start*0 + i] = Ui1[i - start];
        }

        if (s % 10 == 0)
            printTimeRank(sizeProc, Ui1, size, rank);
        Ui.swap(Ui1);
        ;
    }
}

int main(int argc, char** argv)
{
    auto start = std::chrono::high_resolution_clock::now();
    double a = 0, b = 1, t0 = 0, t1 = 0.01;
    double dx = 0.02, dt = 0.0002;

    int Nx = (b-a) / dx;
    int NTimeSteps = (t1-t0) / dt;

    MPI_Init(&argc, &argv);
    int sizeProc = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);

    std::vector<double> U(Nx, 1);

    solveHeat(U, a, b, Nx, rank, sizeProc, dx, dt, NTimeSteps);

    MPI_Finalize();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;

    std::cout << "\nElapsed time for " << sizeProc << " threads = " << duration.count() << std::endl;

    return 0;
}
