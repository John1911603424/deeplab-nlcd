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

#include <gdal.h>

#include "globals.h"
#include "reader.h"
#include "macros.h"

/**
 * The code behind the reader threads.
 *
 * @param _id The id of this particular thread
 * @return Unused
 */
void *reader(void *_id)
{
    uint64_t id = (uint64_t)_id;
    int x_windows = 0;
    int y_windows = 0;
    int slot = -1;
    CPLErr err = CE_None;
    unsigned int state = (unsigned long)id;

    while (operation_mode == training || operation_mode == evaluation)
    {
        // Find an unused data slot
        for (slot = rand_r(&state) % M; (operation_mode == training || operation_mode == evaluation); slot = (slot + 1) % M)
        {
            // If slot is unlocked and slot is empty, read
            if ((pthread_mutex_trylock(&slot_mutexes[slot]) == 0) && (ready[slot] == 0))
            {
                goto read_things;
            }
            UNLOCK_CONTINUE(slot, 100);
        }

        // If search for slot terminated because the mode changed,
        // break out of the loop
        if (operation_mode != training && operation_mode != evaluation)
        {
            UNLOCK_BREAK((slot + M - 1) % M, 0);
        }

    read_things:

#if defined(CHAMPION_EDITION)
        for (int i = 0; i < (1 << 7); ++i)
#else
        for (int i = 0; i < 1; ++i)
#endif
        {
            int wradius = radius / window_size_imagery;

            // Get a suitable training or evaluation window
            if (operation_mode == training) // Training chip
            {
                x_windows = y_windows = -1;
                while (BAD_WINDOW || BAD_TRAINING_WINDOW || EMPTY_WINDOW)
                {
                    const int rand_x = rand_r(&state) % (2 * wradius);
                    const int rand_y = rand_r(&state) % (2 * wradius);
                    x_windows = center_xs[id] + rand_x - wradius;
                    y_windows = center_ys[id] + rand_y - wradius;
                }
            }
            else if (operation_mode == evaluation) // Evaluation chip
            {
                x_windows = y_windows = -1;
                while (BAD_WINDOW || BAD_EVALUATION_WINDOW || EMPTY_WINDOW)
                {
                    const int rand_x = rand_r(&state) % (2 * wradius);
                    const int rand_y = rand_r(&state) % (2 * wradius);
                    x_windows = center_xs[id] + rand_x - wradius;
                    y_windows = center_ys[id] + rand_y - wradius;
                }
            }
            else
            {
                break;
            }

            // Read imagery
            {
                int x = x_windows * window_size_imagery;
                int y = y_windows * window_size_imagery;

                pthread_mutex_lock(&dataset_mutexes[id]);
                err = GDALDatasetRasterIO(imagery_datasets[id], 0,
                                          x, y, window_size_imagery, window_size_imagery,
                                          imagery_slots[slot],
                                          window_size_imagery, window_size_imagery,
                                          imagery_data_type, band_count, bands,
                                          0, 0, 0);
                pthread_mutex_unlock(&dataset_mutexes[id]);
                if (err != CE_None)
                {
                    fprintf(stderr, "FAILED IMAGERY READ AT %d %d\n", x, y);
                    UNLOCK_CONTINUE(slot, 1000)
                }
            }

            // Read labels
            if (label_datasets[id] != NULL)
            {
                int x = x_windows * window_size_labels;
                int y = y_windows * window_size_labels;

                pthread_mutex_lock(&dataset_mutexes[id]);
                err = GDALDatasetRasterIO(label_datasets[id], 0,
                                          x, y, window_size_labels, window_size_labels,
                                          label_slots[slot],
                                          window_size_labels, window_size_labels,
                                          label_data_type, 1, NULL,
                                          0, 0, 0);
                pthread_mutex_unlock(&dataset_mutexes[id]);
                if (err != CE_None)
                {
                    fprintf(stderr, "FAILED LABEL READ AT %d %d\n", x, y);
                    UNLOCK_CONTINUE(slot, 1000)
                }
            }

#if defined(CHAMPION_EDITION)
#define DESIRED (0x02)
            // Imagery and labels have been read, at this point.  Now
            // see if they have the desired characteristics.  If not,
            // continue.
            int should_go_again = 0;

            // Inventory of labels
            uint32_t label_words_found = 0;
            uint64_t num_label_words = 1 * window_size_labels * window_size_labels;
            for (int j = 0; j < num_label_words; ++j)
            {
                uint32_t word = ((uint32_t *)(label_slots[slot]))[j];
                label_words_found |= word;
            }
            if (should_go_again)
            {
                continue;
            }

#if 0
            // Check ensure that IEEE-754 zero is not present in the
            // imagery (the value is forbidden).
            for (int j = 0; j < band_count * window_size_imagery * window_size_imagery; ++j)
            {
                uint32_t word = ((uint32_t *)imagery_slots[slot])[j];
                if (!(word & 0x7fffffff))
                {
                    should_go_again |= 1;
                }
            }
            if (should_go_again)
            {
                continue;
            }
#endif

            // Check to see whether label value 2 was found earlier.
            // If so, this pair is okay.  If not, fall through and
            // (probably) do the for-loop again.
            if (DESIRED & label_words_found)
            {
                break;
            }
#endif
        }

        // The slot is now ready for reading
        ready[slot] = 1;

        // Done
        UNLOCK(slot, 1000)
    }

    return NULL;
}
