# The MIT License (MIT)
# =====================
#
# Copyright © 2020 Azavea
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


def watchdog_thread(seconds):
    """Code for the watchdog thread

    Arguments:
        seconds {int} -- The number of seconds of inactivity to allow before terminating
    """
    while True:
        time.sleep(60)
        if EVALUATIONS_BATCHES_DONE > 0:
            print('EVALUATIONS_DONE={}'.format(EVALUATIONS_BATCHES_DONE))
        with WATCHDOG_MUTEX:
            gap = time.time() - WATCHDOG_TIME
        if gap > seconds:
            print('TERMINATING DUE TO INACTIVITY {} > {}\n'.format(
                gap, seconds), file=sys.stderr)
            os._exit(-1)
