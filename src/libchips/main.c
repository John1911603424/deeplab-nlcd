/*
 * The MIT License (MIT)
 * =====================
 *
 * Copyright © 2019 Azavea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "chips.h"

#define BAND_COUNT (13)

int main(int argc, char **argv)
{
    const int N = 16;
    const int M = 256;
    int L = 1;

    const int window_size = 256;
    int bands[BAND_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    double mus[BAND_COUNT];
    double sigmas[BAND_COUNT];
    float *imagery_buffer = (float *)malloc(window_size * window_size * BAND_COUNT * sizeof(float));
    int32_t *label_buffer = (int32_t *)malloc(window_size * window_size * BAND_COUNT * sizeof(int32_t));

    if (argc > 1)
    {
        sscanf(argv[1], "%d", &L);
    }

    init();
    start_multi(N, M, L,
                "/tmp/mul%d.tif", "/tmp/mask%d.tif",
                6, 5,
                mus, sigmas,
                10000,
                1, window_size, BAND_COUNT, bands);
    fprintf(stderr, "%d %d\n", get_width(0), get_height(0));

    get_statistics("/tmp/mul0.tif", BAND_COUNT, bands, mus, sigmas);
    for (int i = 0; i < BAND_COUNT; ++i)
    {
        fprintf(stderr, "%lf %lf\n", mus[i], sigmas[i]);
    }
    fprintf(stderr, "\n");

    for (int i = 0; i < 1000; ++i)
    {
        get_next(imagery_buffer, label_buffer);
        if (i % 100 == 1)
        {
            recenter(1);
        }

        if (imagery_buffer[0] != 0.0)
        {
            fprintf(stderr, "!");
        }
        else
        {
            fprintf(stderr, ".");
        }
    }
    fprintf(stderr, "\n");

    stop();
    deinit();

    free(imagery_buffer);
    free(label_buffer);

    return 0;
}
