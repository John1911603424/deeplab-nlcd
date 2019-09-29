#ifndef __CHIPS_H__
#define __CHIPS_H__

void init();

void deinit();

void start(int _N,
           int _M,
           const char *imagery_filename, const char *label_filename,
           int _imagery_data_type, int _label_data_type,
           int _operation_mode,
           int _window_size,
           int _band_count, int *_bands);

void stop();

void get_next(void *imagery_buffer, void *label_buffer);

#endif
