#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "chips.h"

int main(int argc, char **argv)
{
    int bands[7] = {1, 2, 3, 4, 5, 6, 7};
    float * imagery_buffer = (float *)malloc(224 * 224 * 7 * sizeof(float));
    int32_t * label_buffer = (int32_t *)malloc(224 * 224 * 7 * sizeof(int32_t));

    start(16, 256, "../../mul.tif", "../../mask.tif", 6, 5, 1, 224, 7, bands);
    get_next(imagery_buffer, label_buffer);
    stop();
    return 0;
}
