#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "chips.h"

int main(int argc, char **argv)
{
    const int band_count = 12;
    const int window_size = 32;
    int bands[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float *imagery_buffer = (float *)malloc(window_size * window_size * band_count * sizeof(float));
    int32_t *label_buffer = (int32_t *)malloc(window_size * window_size * band_count * sizeof(int32_t));

    init();
    start(16, 256, "../../mul.tif", "../../mask.tif", 6, 5, 1, window_size, band_count, bands);
    for (int i = 0; i < 1000; ++i)
    {
        get_next(imagery_buffer, label_buffer);
        fprintf(stderr, "!");
    }
    fprintf(stderr, "\n");
    stop();
    deinit();

    free(imagery_buffer);
    free(label_buffer);

    return 0;
}
