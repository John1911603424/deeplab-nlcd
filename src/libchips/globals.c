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

#include <stdint.h>

#include <pthread.h>

#include <gdal.h>

#include "globals.h"

// Global variables
int N = 0;
int M = 0;
int L = 0;
GDALDataType imagery_data_type = -1;
GDALDataType label_data_type = -1;
int operation_mode = stopped;
int window_size = 0;
int band_count = 0;
int *bands = NULL;
int *widths = NULL;
int *heights = NULL;
int radius = 0;
int *center_xs = NULL;
int *center_ys = NULL;
uint64_t current = 0;

// Thread-related variables
pthread_mutex_t *dataset_mutexes = NULL;
pthread_t *threads = NULL;
GDALDatasetH *imagery_datasets = NULL;
GDALRasterBandH *imagery_first_bands = NULL;
GDALDatasetH *label_datasets = NULL;

// Slot-related variables
pthread_mutex_t *slot_mutexes = NULL;
void **imagery_slots = NULL;
void **label_slots = NULL;
int *ready = NULL;
