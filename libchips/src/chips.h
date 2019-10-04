#ifndef __CHIPS_H__
#define __CHIPS_H__

void init();

void deinit();

int get_width();

int get_height();

int get_inference_chip(void *imagery_buffer,
                       int x_offset, int y_offset,
                       int attempts);

void get_next(void *imagery_buffer, void *label_buffer);

void start(int _N,
           int _M,
           const char *imagery_filename, const char *label_filename,
           int _imagery_data_type, int _label_data_type,
           double * mus, double * sigmas,
           int _operation_mode,
           int _window_size,
           int _band_count, int *_bands);

void stop();

#endif
