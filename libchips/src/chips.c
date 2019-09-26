#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <gdal.h>

// Global variables
int N = 0;
GDALDataType raster_data_type = -1;
GDALDataType label_data_type = -1;
int operation_mode = 0;
int window_size = 0;
int band_count = 0;
int *bands = NULL;
int width = 0;
int height = 0;
int registered = 0;
uint64_t current = 0;

// Thread-related variables
pthread_t *threads = NULL;
pthread_mutex_t *mutexes = NULL;
GDALDatasetH *raster_datasets = NULL;
GDALRasterBandH *raster_bands = NULL;
GDALDatasetH *label_datasets = NULL;
void **raster_arrays = NULL;
void **label_arrays = NULL;
int *ready = NULL;

/**
 * Given a GDAL datat type, return the word length of that type.
 *
 * @param GDALDataType dt The data type
 * @return The word length
 */
int word_size(GDALDataType dt)
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
 * @return The width of the dataset
 */
int get_width()
{
    return width;
}

/**
 * Get the height of the dataset.
 *
 * @return The height of the dataset
 */
int get_height()
{
    return height;
}

/**
 * Get the next available window.
 *
 * @param raster_buffer The return-location for the imagery data
 * @param label_buffer The return-location for the label data
 */
void get_next(void *raster_buffer, void *label_buffer)
{
    for (;; ++current)
    {
        int i = current % N;

        if (!pthread_mutex_trylock(&mutexes[i]))
        {
            if (ready[i] != 1)
            {
                pthread_mutex_unlock(&mutexes[i]);
                continue;
            }
            else
            {
                memcpy(raster_buffer, raster_arrays[i], word_size(raster_data_type) * band_count * window_size * window_size);
                if (label_buffer != NULL)
                {
                    memcpy(label_buffer, label_arrays[i], word_size(label_data_type) * 1 * window_size * window_size);
                }
                ready[i] = 0;
                pthread_mutex_unlock(&mutexes[i]);
                break;
            }
        }
    }
}

/**
 * The code behind the reader threads.
 *
 * @param _i The id of this particular thread
 * @return Unused
 */
void *reader(void *_i)
{
    uint64_t i = (uint64_t)_i;
    int x_offset = 0;
    int y_offset = 0;
    CPLErr err = 0;

    while (operation_mode == 1 || operation_mode == 2)
    {
        // Get training or evaluation chip
        if (operation_mode == 1) // Training chip
        {
            x_offset = y_offset = 0;
            while (((x_offset + y_offset) % 7) == 0)
            {
                x_offset = rand() % (width / window_size);
                y_offset = rand() % (height / window_size);
            }
        }
        else if (operation_mode == 2) // Evaluation chip
        {
            // XXX should not be random
            x_offset = 0;
            y_offset = 1;
            while (((x_offset + y_offset) % 7) != 0)
            {
                x_offset = rand() % (width / window_size);
                y_offset = rand() % (height / window_size);
            }
        }
        x_offset *= window_size;
        y_offset *= window_size;

        // Read data
        pthread_mutex_lock(&mutexes[i]);
        if (ready[i] != 0)
        {
            pthread_mutex_unlock(&mutexes[i]);
            sleep(0);
            continue;
        }
        if (GDAL_DATA_COVERAGE_STATUS_EMPTY & GDALGetDataCoverageStatus(raster_bands[i], x_offset, y_offset, window_size, window_size, 0, NULL))
        {
            pthread_mutex_unlock(&mutexes[i]);
            sleep(0);
            continue;
        }

        err = GDALDatasetRasterIO(raster_datasets[i], 0,
                                  x_offset, y_offset, window_size, window_size,
                                  raster_arrays[i],
                                  window_size, window_size,
                                  raster_data_type, band_count, bands,
                                  0, 0, 0);
        if (err != CE_None)
        {
            pthread_mutex_unlock(&mutexes[i]);
            sleep(0);
            continue;
        }
        err = GDALDatasetRasterIO(label_datasets[i], 0,
                                  x_offset, y_offset, window_size, window_size,
                                  label_arrays[i],
                                  window_size, window_size,
                                  label_data_type, 1, NULL,
                                  0, 0, 0);
        if (err != CE_None)
        {
            pthread_mutex_unlock(&mutexes[i]);
            sleep(0);
            continue;
        }
        ready[i] = 1;
        pthread_mutex_unlock(&mutexes[i]);
    }

    return NULL;
}

/**
 * Given imagery and label filenames, start the reader threads.
 *
 * @param _N The number of reader threads to create
 * @param raster_filename The filename containing the imagery
 * @param label_filename The filename containing the labels
 * @param _operation_mode 1 for training mode, 2 for evaluation mode
 * @param _window_size The desired window size
 * @param _band_count The number of bands
 * @param _bands An array of integers containing the desired bands
 */
void start(int _N,
           const char *raster_filename, const char *label_filename,
           GDALDataType _raster_data_type, GDALDataType _label_data_type,
           int _operation_mode,
           int _window_size,
           int _band_count, int *_bands)
{
    if (!registered)
    {
        srand(time(NULL));
        GDALAllRegister();
        registered = 1;
    }

    // Set globals
    N = _N;
    raster_data_type = _raster_data_type;
    label_data_type = _label_data_type;
    operation_mode = _operation_mode;
    window_size = _window_size;
    band_count = _band_count;
    bands = (int *)malloc(sizeof(int) * band_count);
    memcpy(bands, _bands, sizeof(int) * band_count);

    // Create arrays
    raster_datasets = (GDALDatasetH *)malloc(sizeof(GDALDatasetH) * N);
    raster_bands = (GDALRasterBandH *)malloc(sizeof(GDALRasterBandH) * N);
    label_datasets = (GDALDatasetH *)malloc(sizeof(GDALDatasetH) * N);
    raster_datasets[0] = GDALOpen(raster_filename, GA_ReadOnly);
    width = GDALGetRasterXSize(raster_datasets[0]);
    height = GDALGetRasterYSize(raster_datasets[0]);
    threads = (pthread_t *)malloc(sizeof(pthread_t) * N);
    mutexes = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * N);
    raster_arrays = malloc(sizeof(void *) * N);
    label_arrays = malloc(sizeof(void *) * N);
    ready = calloc(N, sizeof(int));

    // Fill arrays, start threads
    for (int64_t i = 0; i < N; ++i)
    {
        if (i != 0)
            raster_datasets[i] = GDALOpen(raster_filename, GA_ReadOnly);
        raster_bands[i] = GDALGetRasterBand(raster_datasets[i], 1);
        label_datasets[i] = GDALOpen(label_filename, GA_ReadOnly);
        pthread_mutex_init(&mutexes[i], NULL);
        raster_arrays[i] = malloc(word_size(raster_data_type) * band_count * window_size * window_size);
        label_arrays[i] = malloc(word_size(label_data_type) * 1 * window_size * window_size);
        threads[i] = pthread_create(&threads[i], NULL, reader, (void *)i);
    }

    return;
}

/**
 * Stop the reader threads.
 */
void stop()
{
    operation_mode = 0;
    for (int i = 0; i < N; ++i)
    {
        pthread_join(threads[i], NULL);
        GDALClose(raster_datasets[i]);
        GDALClose(label_datasets[i]);
        free(raster_arrays[i]);
        free(label_arrays[i]);
        ready[i] = 0;
    }

    free(bands);
    free(threads);
    free(mutexes);
    free(raster_datasets);
    free(raster_bands);
    free(label_datasets);
    free(raster_arrays);
    free(label_arrays);
    free(ready);

    bands = NULL;
    threads = NULL;
    mutexes = NULL;
    raster_datasets = NULL;
    raster_bands = NULL;
    label_datasets = NULL;
    raster_arrays = NULL;
    label_arrays = NULL;
    ready = NULL;
}
