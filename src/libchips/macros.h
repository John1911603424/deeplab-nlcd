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

#ifndef __MACROS_H__
#define __MACROS_H__

#define unlikely(x) __builtin_expect(!!(x), 0)

#define UNLOCK(index, useconds)                     \
    {                                               \
        pthread_mutex_unlock(&slot_mutexes[index]); \
        usleep(useconds);                           \
    }
#define UNLOCK_BREAK(index, useconds) \
    {                                 \
        UNLOCK(index, useconds)       \
        break;                        \
    }
#define UNLOCK_CONTINUE(index, useconds) \
    {                                    \
        UNLOCK(index, useconds)          \
        continue;                        \
    }

#define EMPTY_WINDOW (GDAL_DATA_COVERAGE_STATUS_EMPTY & GDALGetDataCoverageStatus(           \
                                                            imagery_first_bands[id],         \
                                                            window_size_imagery * x_windows, \
                                                            window_size_imagery * y_windows, \
                                                            window_size_imagery,             \
                                                            window_size_imagery,             \
                                                            0, NULL))
#define BAD_WINDOW ((x_windows < 0 || x_windows > ((widths[id] / window_size_imagery) - 1) || \
                     (y_windows < 0 || y_windows > ((heights[id] / window_size_imagery) - 1))))
#define BAD_TRAINING_WINDOW (((x_windows + y_windows) % 7) == 0)
#define BAD_EVALUATION_WINDOW (((x_windows + y_windows) % 7) != 0)

#endif
