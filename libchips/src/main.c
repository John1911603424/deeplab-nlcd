#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "chips.h"

#define BAND_COUNT (12)

int main(int argc, char **argv)
{
    const int window_size = 256;
    int bands[BAND_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double mus[BAND_COUNT];
    double sigmas[BAND_COUNT];
    float *imagery_buffer = (float *)malloc(window_size * window_size * BAND_COUNT * sizeof(float));
    int32_t *label_buffer = (int32_t *)malloc(window_size * window_size * BAND_COUNT * sizeof(int32_t));

    init();
    start(16, 256,
          "../../mul.tif", "../../mask.tif",
          6, 5,
          mus, sigmas,
          1, window_size, BAND_COUNT, bands);
    for (int i = 0; i < BAND_COUNT; ++i)
    {
        fprintf(stderr, "%lf %lf\n", mus[i], sigmas[i]);
    }
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
