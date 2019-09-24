#include <stdio.h>
#include <gdal.h>

int moop(double *array)
{
    for (int i = 0; i < 9; ++i)
    {
        array[i] = i;
    }
    return 0;
}
