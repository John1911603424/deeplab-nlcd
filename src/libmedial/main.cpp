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

#include <cstdio>
#include <cmath>

#include <string>
#include <fstream>
#include <streambuf>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "medial.h"

namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<int64_t> polygon_integral_point;
typedef bg::model::referring_segment<polygon_integral_point> polygon_segment;
typedef bg::model::polygon<polygon_integral_point> polygon;

int main(int argc, char **argv)
{
    // https://upload.wikimedia.org/wikipedia/commons/5/55/SFA_Polygon_with_hole.svg
    const int f = 1000;
    polygon bg_polygon;
    std::vector<int64_t> segments;

    std::ifstream t(argv[1]);
    std::string wkt_polygon = std::string(std::istreambuf_iterator<char>(t),
                                          std::istreambuf_iterator<char>());

    int n = -1;
    double *return_data;

    bg::read_wkt(wkt_polygon.data(), bg_polygon);
    bg::for_each_segment(bg_polygon, [&segments](polygon_segment s) {
        segments.push_back(s.first.x() * f);
        segments.push_back(s.first.y() * f);
        segments.push_back(s.second.x() * f);
        segments.push_back(s.second.y() * f);
    });

    // Compute the internal axis
    n = get_skeleton(segments.size(), segments.data(), &return_data);
    fprintf(stderr, "n = %d\n", n);

    // Display the internal axis
    for (int i = 0; i < n; i += 6)
    {
        double x1, y1, d1, x2, y2, d2, length;

        x1 = return_data[i + 0] / f;
        y1 = return_data[i + 1] / f;
        x2 = return_data[i + 2] / f;
        y2 = return_data[i + 3] / f;
        d1 = return_data[i + 4] / f;
        d2 = return_data[i + 5] / f;
        length = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

        fprintf(stderr, "INTERNAL EDGE: (%lf %lf) (%lf %lf) with distances %lf %lf and length %lf\n", x1, y1, x2, y2, d1, d2, length);
    }

    // Cleanup
    free(return_data);
    return_data = nullptr;

    return 0;
}
