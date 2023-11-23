#include <CL/sycl.hpp>
#include <stdio.h>

#define BLOCK_SIZE 128

int rows = 4, cols = 4;
int *data;
int wall[4][4] = {{3, 0, 7, 5}, {6, 5, 4, 2}, {3, 8, 3, 2}, {4, 6, 0, 1}};
int *result = new int[cols];
int pyramid_height = 1;

#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

void dynproc_kernel(int *gpuSrc,
                    int *gpuWall,
                    int cols,
                    int iteration,
                    int cur_row,
                    int *gpuResult,
                    sycl::nd_item<3> item_ct1)
{

    int current_element = item_ct1.get_group(0);
    int left_ind = current_element - 1 ? current_element >= 1 : 0;
    int right_ind = current_element + 1 ? current_element < cols : cols;
    int up_ind = current_element;

    for (int i = 0; i < iteration; i++) {
        int index = (cur_row + i) * cols + current_element;
        int left = gpuSrc[left_ind];
        int up = gpuSrc[up_ind];
        int right = gpuSrc[right_ind];

        int shortest = MIN(left, up);
        shortest = MIN(shortest, right);
        item_ct1.barrier();
        gpuSrc[current_element] = gpuWall[index] + shortest;
    }

    item_ct1.barrier();
    gpuResult[current_element] = gpuSrc[current_element];
}

void run()
{

    int rows_val = rows, cols_val = cols, pyramid_height_val = pyramid_height;

    int *gpuWall, *gpuResult[2];
    int size = rows * cols;

    sycl::queue defaultQueue;

    gpuResult[0] = sycl::malloc_device<int>(cols_val, defaultQueue);
    gpuResult[1] = sycl::malloc_device<int>(cols_val, defaultQueue);
    gpuWall = sycl::malloc_device<int>((size - cols_val), defaultQueue);

    defaultQueue.memcpy(gpuResult[0], data, sizeof(int) * cols_val).wait();

    defaultQueue
        .memcpy(gpuWall, data + cols_val, sizeof(int) * (size - cols_val))
        .wait();

    sycl::range<3> dimBlock(1, 1, cols);
    sycl::range<3> dimGrid(1, 1, cols);

    for (int t = 0; t < rows; t += pyramid_height) {

        int iteration = MIN(pyramid_height_val, rows_val - t - 1);
        defaultQueue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(dimBlock * dimGrid, dimGrid),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dynproc_kernel(gpuResult[0], gpuWall, cols_val,
                                                iteration, t - 1, gpuResult[1],
                                                item_ct1);
                             });
            defaultQueue.memcpy(gpuResult[0], gpuResult[1], sizeof(int) * cols)
                .wait();
        });
    }

    defaultQueue.memcpy(result, gpuResult[1], sizeof(int) * cols).wait();

    for (int i = 0; i < cols; i++)
        printf("%d ", result[i]);
}

int main(int argc, char **argv)
{
    run();
    return EXIT_SUCCESS;
}
