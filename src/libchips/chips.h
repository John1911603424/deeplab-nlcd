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

#ifndef __CHIPS_H__
#define __CHIPS_H__

void init();

void deinit();

int get_width();

int get_height();

void get_statistics(const char *imagery_filename,
                    int band_count,
                    int *bands,
                    double *mus,
                    double *sigmas);

int get_inference_chip(void *imagery_buffer,
                       int x_offset, int y_offset,
                       int attempts);

void recenter(int verbose);

void get_next(void *imagery_buffer, void *label_buffer);

void start(int _N,
           int _M,
           const char *imagery_filename, const char *label_filename,
           int _imagery_data_type, int _label_data_type,
           double *mus, double *sigmas,
           int _radius,
           int _operation_mode,
           int _window_size,
           int _band_count, int *_bands);

void start_multi(int _N,
                 int _M,
                 int _L,
                 const char *imagery_filename, const char *label_filename,
                 int _imagery_data_type, int _label_data_type,
                 double *mus, double *sigmas,
                 int _radius,
                 int _operation_mode,
                 int _window_size,
                 int _band_count, int *_bands);

void stop();

#endif
