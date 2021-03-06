/*
 * The MIT License (MIT)
 * =====================
 *
 * Copyright © 2019 Azavea
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
#include <cstdint>

#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::d2::point_xy<double> point_t;
typedef bg::model::box<point_t> box_t;
typedef bg::model::polygon<point_t> polygon_t;
typedef bg::model::multi_polygon<polygon_t> multipolygon_t;
typedef std::pair<box_t, int> value_t;
typedef bgi::rtree<value_t, bgi::linear<8>> rtree_t;

std::vector<rtree_t> trees;

extern "C" int add_tree()
{
    trees.emplace_back();
    return trees.size();
}

extern "C" double query(int index, double xmin, double ymin, double xmax, double ymax)
{
    box_t box = box_t(point_t(xmin, ymin), point_t(xmax, ymax));
    multipolygon_t difference;
    bg::convert(box, difference);
    std::vector<value_t> results;

    trees[index].query(bgi::intersects(box), std::back_inserter(results));

    for (const auto &result : results)
    {
        multipolygon_t new_difference;
        bg::difference(difference, result.first, new_difference);
        difference = new_difference;
    }

    return (bg::area(difference) / bg::area(box));
}

extern "C" void insert(int index, double xmin, double ymin, double xmax, double ymax)
{
    box_t box = box_t(point_t(xmin, ymin), point_t(xmax, ymax));
    trees[index].insert(std::make_pair(box, 33));
}
