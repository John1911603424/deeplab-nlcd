/*
 * The MIT License (MIT)
 * =====================
 *
 * Copyright © 2019-2020 Azavea
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

#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include <pthread.h>

#include <gdal.h>

enum op_mode
{
    stopped = 0,
    training = 1,
    evaluation = 2,
    inference = 3,
};

// Global variables
extern int N;
extern int M;
extern int L;
extern GDALDataType imagery_data_type;
extern GDALDataType label_data_type;
extern int operation_mode;
extern int window_size_imagery;
extern int window_size_labels;
extern int band_count;
extern int *bands;
extern int *widths;
extern int *heights;
extern int radius;
extern int *center_xs;
extern int *center_ys;
extern uint64_t current;

// Thread-related variables
extern pthread_mutex_t *dataset_mutexes;
extern pthread_t *threads;
extern GDALDatasetH *imagery_datasets;
extern GDALRasterBandH *imagery_first_bands;
extern GDALDatasetH *label_datasets;

// Slot-related variables
extern pthread_mutex_t *slot_mutexes;
extern void **imagery_slots;
extern void **label_slots;
extern int *ready;

#endif
