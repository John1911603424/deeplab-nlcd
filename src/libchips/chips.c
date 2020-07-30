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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <assert.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

#include <gdal.h>

#include "globals.h"
#include "reader.h"
#include "macros.h"

/**
 * Initialize the library.
 */
void init()
{
    GDALAllRegister();
}

/**
 * Deinitialize the library.
 */
void deinit()
{
    GDALDestroy();
}

/**
 * Given a GDAL data type, return the word length of that type.
 *
 * @param GDALDataType dt The data type
 * @return The word length
 */
static int word_size(GDALDataType dt)
{
    switch (dt)
    {
    case GDT_Byte:
        return 1;
    case GDT_UInt16:
        return 2;
    case GDT_Int16:
        return 2;
    case GDT_UInt32:
        return 4;
    case GDT_Int32:
        return 4;
    case GDT_Float32:
        return 4;
    case GDT_Float64:
        return 8;
    case GDT_CInt16:
        return 4;
    case GDT_CInt32:
        return 8;
    case GDT_CFloat32:
        return 8;
    case GDT_CFloat64:
        return 16;
    default:
        assert(0);
    }
}

/**
 * Get the width of the dataset.
 *
 * @param index The index of the dataset in question
 * @return The width of the dataset
 */
int get_width(int index)
{
    return widths[index];
}

/**
 * Get the height of the dataset.
 *
 * @param index The index of the dataset in question
 * @return The height of the dataset
 */
int get_height(int index)
{
    return heights[index];
}

/**
 * Get statistics from an image.
 *
 * @param imagery_filename The imagery from which to get the statistics
 * @param band_count The number of bands
 * @param mus The return-location of the means
 * @param sigmas The return-location of the sigmas
 */
void get_statistics(const char *imagery_filename,
                    int band_count,
                    int *bands,
                    double *mus,
                    double *sigmas)
{
    GDALDatasetH dataset;

    dataset = GDALOpen(imagery_filename, GA_ReadOnly);
    for (int i = 0; mus && sigmas && (i < band_count); ++i)
    {
        GDALRasterBandH band = GDALGetRasterBand(dataset, bands[i]);
        GDALGetRasterStatistics(band, 1, 1, NULL, NULL, mus + i, sigmas + i);
    }
    GDALClose(dataset);
    return;
}

/**
 * Get an (inference) chip.  This can be used only if operation_mode 3 (inference)
 * is active.
 *
 * @param imagery_buffer The return-pointer for the imagery data
 * @param x_offset The x-offset for the window (in pixels)
 * @param y_offset The y-offset for the window (in pixels)
 * @param attempts The maximum number of attempts to make to read the window
 * @return 1 for success, 0 for failure
 */
int get_inference_chip(void *imagery_buffer,
                       int x, int y,
                       int attempts)
{
    int id = 0;
    int x_offset = x / window_size;
    int y_offset = y / window_size;

    if ((operation_mode != inference) || EMPTY_WINDOW)
    {
        memset(imagery_buffer, 0, word_size(imagery_data_type) * band_count * window_size * window_size);
        return 0;
    }

    for (int i = 0; i < attempts; ++i)
    {
        CPLErr err = CE_None;

        // Read imagery
        err = GDALDatasetRasterIO(imagery_datasets[id], 0,
                                  x, y, window_size, window_size,
                                  imagery_buffer,
                                  window_size, window_size,
                                  imagery_data_type, band_count, bands,
                                  0, 0, 0);
        if (err != CE_None)
        {
            continue;
        }
        else
        {
            break;
        }
    }

    return 1;
}

/**
 * Recenter
 */
void recenter(int verbose)
{
    for (int id = 0; id < N; ++id)
    {
        struct timespec tp;
        clock_gettime(CLOCK_REALTIME, &tp);
        int x_offset = -1;
        int y_offset = -1;

        pthread_mutex_lock(&dataset_mutexes[id]);
        while (BAD_WINDOW || EMPTY_WINDOW)
        {
            x_offset = rand_r((unsigned int *)&tp.tv_nsec) % (widths[id] / window_size);
            y_offset = rand_r((unsigned int *)&tp.tv_nsec) % (heights[id] / window_size);
        }
        center_xs[id] = x_offset;
        center_ys[id] = y_offset;
        pthread_mutex_unlock(&dataset_mutexes[id]);
    }

    if (verbose)
    {
        fprintf(stderr, "RECENTERED:");
        for (int id = 0; id < N; ++id)
        {
            fprintf(stderr, " {id = %d: x = %d y = %d}", id, center_xs[id] * window_size, center_ys[id] * window_size);
        }
        fprintf(stderr, "\n");
    }
}

/**
 * Get the next available window.
 *
 * @param imagery_buffer The return-location for the imagery data
 * @param label_buffer The return-location for the label data
 */
void get_next(void *imagery_buffer, void *label_buffer)
{
    for (;; ++current)
    {
        int slot = current % M;

        if (pthread_mutex_trylock(&slot_mutexes[slot]) == 0)
        {
            if (ready[slot] != 1)
            {
                pthread_mutex_unlock(&slot_mutexes[slot]);
                continue;
            }
            else if (ready[slot] == 1)
            {
                memcpy(imagery_buffer, imagery_slots[slot], word_size(imagery_data_type) * band_count * window_size * window_size);
                if (label_buffer != NULL)
                {
                    memcpy(label_buffer, label_slots[slot], word_size(label_data_type) * 1 * window_size * window_size);
                }
                ready[slot] = 0;
                pthread_mutex_unlock(&slot_mutexes[slot]);
                break;
            }
        }
    }
}

/**
 * Given imagery and label filenames, start the reader threads.
 *
 * @param _N The number of reader threads to create
 * @param _M The number of slots
 * @param _L The number of imagery, label pairs
 * @param imagery_filename_template The filename template for the imagery
 * @param label_filename_template The filename template for the labels
 * @param mus Return-location for the (approximate) means of the bands
 * @param sigmas return-location for the (approximate) standard deviations of the bands
 * @param _radius The approximate radius (in pixels) of the typical component of the image
 * @param _operation_mode 1 for training mode, 2 for evaluation mode, 3 for inference mode
 * @param _window_size The desired window size
 * @param _band_count The number of bands
 * @param _bands An array of integers containing the desired bands
 */
void start(int _N,
           int _M,
           int _L,
           const char *imagery_filename_template,
           const char *label_filename_template,
           GDALDataType _imagery_data_type, GDALDataType _label_data_type,
           double *mus, double *sigmas,
           int _radius,
           int _operation_mode,
           int _window_size,
           int _band_count, int *_bands)
{
    // Set globals
    N = _N;
    M = _M;
    L = _L;
    imagery_data_type = _imagery_data_type;
    label_data_type = _label_data_type;
    operation_mode = _operation_mode;
    window_size = _window_size;
    band_count = _band_count;
    bands = (int *)malloc(sizeof(int) * band_count);
    radius = _radius;
    memcpy(bands, _bands, sizeof(int) * band_count);

    // Per-thread arrays (except for width and height)
    imagery_datasets = (GDALDatasetH *)malloc(sizeof(GDALDatasetH) * N);
    imagery_first_bands = (GDALRasterBandH *)malloc(sizeof(GDALRasterBandH) * N);
    label_datasets = (GDALDatasetH *)malloc(sizeof(GDALDatasetH) * N);
    widths = (int *)malloc(sizeof(int) * N);
    heights = (int *)malloc(sizeof(int) * N);
    center_xs = (int *)malloc(sizeof(int) * N);
    center_ys = (int *)malloc(sizeof(int) * N);
    {
        char imagery_filename[0xff];

        sprintf(imagery_filename, imagery_filename_template, 0);
        imagery_datasets[0] = GDALOpen(imagery_filename, GA_ReadOnly);
        imagery_first_bands[0] = GDALGetRasterBand(imagery_datasets[0], 1);
    }
    widths[0] = GDALGetRasterXSize(imagery_datasets[0]);
    heights[0] = GDALGetRasterYSize(imagery_datasets[0]);
    threads = (pthread_t *)malloc(sizeof(pthread_t) * N);
    dataset_mutexes = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * N);

    // Per-slot arrays
    slot_mutexes = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * M);
    imagery_slots = malloc(sizeof(void *) * M);
    label_slots = malloc(sizeof(void *) * M);
    ready = calloc(M, sizeof(int));

    // Fill arrays
    for (int64_t i = 0; i < M; ++i)
    {
        imagery_slots[i] = malloc(word_size(imagery_data_type) * band_count * window_size * window_size);
        label_slots[i] = malloc(word_size(label_data_type) * 1 * window_size * window_size);
        pthread_mutex_init(&slot_mutexes[i], NULL);
    }
    for (int i = 0; mus && sigmas && (i < band_count); ++i)
    {
        GDALRasterBandH band = GDALGetRasterBand(imagery_datasets[0], bands[i]);
        GDALGetRasterStatistics(band, 1, 1, NULL, NULL, mus + i, sigmas + i);
    }
    for (int64_t i = 0; i < N; ++i)
    {
        char imagery_or_label_filename[0xff];

        pthread_mutex_init(&dataset_mutexes[i], NULL);
        if (i != 0)
        {
            sprintf(imagery_or_label_filename, imagery_filename_template, i % L);
            imagery_datasets[i] = GDALOpen(imagery_or_label_filename, GA_ReadOnly);
            imagery_first_bands[i] = GDALGetRasterBand(imagery_datasets[i], 1);
            widths[i] = GDALGetRasterXSize(imagery_datasets[i]);
            heights[i] = GDALGetRasterYSize(imagery_datasets[i]);
        }
        if (label_filename_template != NULL)
        {
            sprintf(imagery_or_label_filename, label_filename_template, i % L);
            label_datasets[i] = GDALOpen(imagery_or_label_filename, GA_ReadOnly);
        }
        else
        {
            label_datasets[i] = NULL;
        }
    }

    // Start threads
    recenter(0);
    for (int64_t i = 0; i < N; ++i)
    {
        pthread_create(&threads[i], NULL, reader, (void *)i);
    }

    return;
}

/**
 * Stop the reader threads.
 */
void stop()
{
    operation_mode = stopped;
    for (int i = 0; i < N; ++i)
    {
        pthread_join(threads[i], NULL);
        GDALClose(imagery_datasets[i]);
        if (label_datasets[i] != NULL)
        {
            GDALClose(label_datasets[i]);
        }
    }
    for (int i = 0; i < M; ++i)
    {
        free(imagery_slots[i]);
        free(label_slots[i]);
        ready[i] = 0;
    }

    free(bands);
    free(threads);
    free(slot_mutexes);
    free(dataset_mutexes);
    free(imagery_datasets);
    free(imagery_first_bands);
    free(label_datasets);
    free(imagery_slots);
    free(label_slots);
    free(ready);
    free(widths);
    free(heights);
    free(center_xs);
    free(center_ys);

    N = M = L = 0;
    bands = NULL;
    threads = NULL;
    slot_mutexes = NULL;
    dataset_mutexes = NULL;
    imagery_datasets = NULL;
    imagery_first_bands = NULL;
    label_datasets = NULL;
    imagery_slots = NULL;
    label_slots = NULL;
    ready = NULL;
}
