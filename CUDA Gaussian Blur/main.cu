%%cu
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>

void make_filter(float* filter, const int fSize, const float sigma) {
    int center = (fSize)/2;
    double s = 2 * sigma;
    double sum = 0.0;
    for (int i = -center; i <= center; ++i) {
        for (int j = -center; j <= center; ++j) {
            double r = sqrt(i * i + j * j);
            auto res = (exp(-(r * r) / s)) / (M_PI * s);
            filter[(i + center) * fSize + (j + center)] = res;
            sum += res;
        }
    }
}

void print_channels(FILE *fo, char name, const float *ch, int Nrows, int Ncols) {
    fprintf(fo, "%c\n", name);
    for (size_t i = 0; i < Nrows; ++i) {
        for (size_t j = 0; j < Ncols; ++j) {
            int idx = i * Ncols + j;
            fprintf(fo, "%f ", ch[idx]);
        }
        fprintf(fo, "\n");
    }
    fprintf(fo, "\n");
}

__global__ void gaussian_blur(const float *inputChannel, float* blurredChannel, int Nrows, int Ncols,
                              const float* const filter, const int fSize) {

    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= Ncols || row >= Nrows) {
        return;
    }

    float c = 0.0f;

    for (int fCol = 0; fCol < fSize; fCol++) {
        for (int fRow = 0; fRow < fSize; fRow++) {

            int imageCol = column + fCol - fSize / 2;
            int imageRow = row + fRow - fSize / 2;

            imageCol = min(max(imageCol,0),Ncols-1);
            imageRow = min(max(imageRow,0),Nrows-1);

            c += (filter[fRow * fSize + fCol] * inputChannel[imageRow*Ncols+imageCol]);
        }
    }

    blurredChannel[row * Ncols + column] = c;
}

void worker(FILE *fo, float sigma) {
    FILE *file;
    file = fopen("case.txt", "r");

    int Nrows, Ncols;
    if (fscanf(file, "%d%d", &Nrows, &Ncols) == EOF) {
        return;
    }

    int Npixels = (int)Nrows * (int)Ncols;
    size_t size = Npixels * (sizeof(float));

    float *h_r_in = (float *)malloc(size);
    float *h_g_in = (float *)malloc(size);
    float *h_b_in = (float *)malloc(size);

    // сделать в функции, учитывая указатели
    for (size_t i = 0; i < Nrows; ++i) {
      for (size_t j = 0; j < Ncols; ++j) {
         fscanf(file, "%f", &h_r_in[i * Ncols + j]);
      }
    }

    for (size_t i = 0; i < Nrows; ++i) {
      for (size_t j = 0; j < Ncols; ++j) {
         fscanf(file, "%f", &h_g_in[i * Ncols + j]);
      }
    }

    for (size_t i = 0; i < Nrows; ++i) {
      for (size_t j = 0; j < Ncols; ++j) {
         fscanf(file, "%f", &h_b_in[i * Ncols + j]);
      }
    }

    fclose(file);

    int fSize = 25;
    size_t size_filter = fSize * fSize * sizeof(float);

    printf("SIGMA = %f;\t", sigma);
    float *h_filter = (float *)malloc(size_filter);
    sigma *= sigma;
    make_filter(h_filter, fSize, sigma);

    float *d_r_in;
    cudaMalloc(&d_r_in, size);
    cudaMemcpy(d_r_in, h_r_in, size, cudaMemcpyHostToDevice);

    float *d_g_in;
    cudaMalloc(&d_g_in, size);
    cudaMemcpy(d_g_in, h_g_in, size, cudaMemcpyHostToDevice);

    float *d_b_in;
    cudaMalloc(&d_b_in, size);
    cudaMemcpy(d_b_in, h_b_in, size, cudaMemcpyHostToDevice);

    float *d_filter;
    cudaMalloc(&d_filter, size_filter);
    cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);

    float *d_r_out;
    cudaMalloc(&d_r_out, size);
    float *d_g_out;
    cudaMalloc(&d_g_out, size);
    float *d_b_out;
    cudaMalloc(&d_b_out, size);

    const dim3 blockSize(1, 1, 1);
    const dim3 gridSize(Ncols / blockSize.x + 1, Nrows / blockSize.y + 1, 1);

    float elapsed=0;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    gaussian_blur<<<gridSize, blockSize>>>(d_r_in, d_r_out, Nrows, Ncols, d_filter, fSize);
    gaussian_blur<<<gridSize, blockSize>>>(d_g_in, d_g_out, Nrows, Ncols, d_filter, fSize);
    gaussian_blur<<<gridSize, blockSize>>>(d_b_in, d_b_out, Nrows, Ncols, d_filter, fSize);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Elapsed time %.2f ms\n", elapsed);

    float *h_r_out = (float *)malloc(size);
    cudaMemcpy(h_r_out, d_r_out, size, cudaMemcpyDeviceToHost);
    float *h_g_out = (float *)malloc(size);
    cudaMemcpy(h_g_out, d_g_out, size, cudaMemcpyDeviceToHost);
    float *h_b_out = (float *)malloc(size);
    cudaMemcpy(h_b_out, d_b_out, size, cudaMemcpyDeviceToHost);

    print_channels(fo, 'R', h_r_out, Nrows, Ncols);
    print_channels(fo, 'G', h_g_out, Nrows, Ncols);
    print_channels(fo, 'B', h_b_out, Nrows, Ncols);

    cudaFree(d_filter);
    free(h_filter);

    cudaFree(d_r_in);
    cudaFree(d_r_out);
    cudaFree(h_r_out);
    free(h_r_out);

    cudaFree(d_g_in);
    cudaFree(d_g_out);
    cudaFree(h_g_out);
    free(h_g_out);

    cudaFree(d_b_in);
    cudaFree(d_b_out);
    cudaFree(h_b_out);
    free(h_b_out);

    free(h_r_in);
    free(h_g_in);
    free(h_b_in);
}

int main() {
    
    FILE *fo;
    fo = fopen("tree_25_4_1.txt", "w");

    worker(fo, 4);
    fclose(fo);

    return 0;
}
