# -*- coding: utf-8 -*-

"""
/***************************************************************************
 Pathfinder
                                 A QGIS plugin
 Finds near-optimal paths in raster images
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-01-15
        copyright            : (C) 2022 by Noah Mollerstuen
        email                : noah@mollerstuen.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Noah Mollerstuen'
__date__ = '2022-01-15'
__copyright__ = '(C) 2022 by Noah Mollerstuen'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import typing as t
import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProject,
                       QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterPoint,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFeatureSink,
                       QgsWkbTypes,
                       QgsFields,
                       QgsFeature,
                       QgsGeometry,
                       QgsPoint,
                       QgsRectangle)
from osgeo import gdal
import heapq

from .formulas import parse_formula, evaluate_formula

DIRECTION_MAPPING = {
    (0, 1): 1,  # 0 is reserved for None
    (1, 0): 2,
    (0, -1): 3,
    (-1, 0): 4
}
DIRECTION_MAPPING_INV = {v: k for k, v in DIRECTION_MAPPING.items()}
NEIGHBORS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def point_to_pixel(point: QgsPoint, img_bounds: QgsRectangle, img_width: int, img_height: int) -> (int, int):
    return (
        round((point.x() - img_bounds.xMinimum()) / img_bounds.width() * img_width - 0.5),
        img_height - 1 - round((point.y() - img_bounds.yMinimum()) / img_bounds.height() * img_height - 0.5)
    )


def pixel_to_point(pix: (int, int), img_bounds: QgsRectangle, img_width: int, img_height: int) -> QgsPoint:
    return QgsPoint(
        (pix[0] + 0.5) / img_width * img_bounds.width() + img_bounds.xMinimum(),
        (img_height - 1 - pix[1] + 0.5) / img_height * img_bounds.height() + img_bounds.yMinimum()
    )


def a_star_heuristic(pos1: (int, int), pos2: (int, int)) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(pos: (int, int), max_x, max_y):
    return [
        p for p in (((pos[0] + n[0], pos[1] + n[1]), n) for n in NEIGHBORS) if
        0 <= p[0][0] < max_x and 0 <= p[0][1] < max_y
    ]


vars_dict = {}


def eval_expression_at_pos(formula: str, pos: (int, int), inp_arrays, parsed_formula=None):
    global vars_dict

    x = pos[0]
    y = pos[1]
    vars_dict["x"] = x
    vars_dict["y"] = y

    for i, arr in enumerate(inp_arrays):
        s = str(i + 1)
        vars_dict["val" + s] = arr[y][x]

    return evaluate_formula(formula, vars_dict, parsed_formula)


class PriorityQueue:
    def __init__(self):
        self.elements: t.List[t.Tuple[float, (int, int)]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: (int, int), priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> (int, int):
        return heapq.heappop(self.elements)[1]


class PathfinderAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.
    OUTPUT = 'OUTPUT'

    INPUT_IMAGES = ["INPUT_IMG" + str(i) for i in range(3)]
    INPUT_POINT1 = 'INPUT_POINT1'
    INPUT_POINT2 = 'INPUT_POINT2'
    INPUT_MIN_VAL = 'INPUT_MIN_VAL'
    INPUT_MAX_VAL = 'INPUT_MAX_VAL'
    INPUT_TRAVERSABILITY_EXPRESSION = 'INPUT_TRAVERSABILITY_EXPRESSION'
    INPUT_COST_EXPRESSION = 'INPUT_COST_EXPRESSION'

    def initAlgorithm(self, config):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_IMAGES[0],
                self.tr('Primary Input Image')
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_IMAGES[1],
                self.tr('Secondary Input Image'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_IMAGES[2],
                self.tr('Tertiary Input Image'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterPoint(
                self.INPUT_POINT1,
                self.tr('Starting Point')
            )
        )

        self.addParameter(
            QgsProcessingParameterPoint(
                self.INPUT_POINT2,
                self.tr('Ending Point')
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_MIN_VAL,
                self.tr('Minimum Traversable Value'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_MAX_VAL,
                self.tr('Maximum Traversable Value'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.INPUT_TRAVERSABILITY_EXPRESSION,
                self.tr('Custom Traversability Expression'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.INPUT_COST_EXPRESSION,
                self.tr('Custom Cost Expression'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        inp_layers = []
        for code in self.INPUT_IMAGES:
            param = self.parameterAsRasterLayer(parameters, code, context)
            if param is not None:
                inp_layers.append(param)

        start_point = self.parameterAsPoint(parameters, self.INPUT_POINT1, context)
        end_point = self.parameterAsPoint(parameters, self.INPUT_POINT2, context)

        min_val = self.parameterAsString(parameters, self.INPUT_MIN_VAL, context)
        try:
            min_val = float(min_val)
        except ValueError:
            min_val = None
        max_val = self.parameterAsString(parameters, self.INPUT_MAX_VAL, context)
        try:
            max_val = float(max_val)
        except ValueError:
            max_val = None

        cost_expression_str = self.parameterAsString(parameters, self.INPUT_COST_EXPRESSION, context)
        if cost_expression_str == "":
            cost_expression = None
        else:
            cost_expression = parse_formula(cost_expression_str)

        traversability_expression_str = self.parameterAsString(
            parameters, self.INPUT_TRAVERSABILITY_EXPRESSION, context)
        if traversability_expression_str == "":
            traversability_expression = None
        else:
            traversability_expression = parse_formula(traversability_expression_str)

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, QgsFields(),
                                               geometryType=QgsWkbTypes.Type.LineString, crs=inp_layers[0].crs())

        bounding_rect = inp_layers[0].extent()

        if not bounding_rect.contains(start_point):
            raise ValueError(self.tr("Starting Point must be somewhere within the first raster image"))
        if not bounding_rect.contains(end_point):
            raise ValueError(self.tr("Ending Point must be somewhere within the first raster image"))

        inp_arrs: t.List[np.ndarray] = []
        for layer in inp_layers:
            img_ds = gdal.Open(layer.dataProvider().dataSourceUri())
            inp_arrs.append(img_ds.GetRasterBand(1).ReadAsArray())

        start_pos = point_to_pixel(start_point, bounding_rect, inp_arrs[0].shape[1], inp_arrs[0].shape[0])
        if (min_val is not None and inp_arrs[0][start_pos[1]][start_pos[0]] < min_val) \
                or (max_val is not None and inp_arrs[0][start_pos[1]][start_pos[0]] > max_val) \
                or (traversability_expression is not None and not eval_expression_at_pos(traversability_expression_str,
                                                                                         start_pos, inp_arrs,
                                                                                         traversability_expression)):
            raise ValueError(self.tr("Starting Point must be in a traversable area"))
        end_pos = point_to_pixel(end_point, bounding_rect, inp_arrs[0].shape[1], inp_arrs[0].shape[0])
        if (min_val is not None and inp_arrs[0][end_pos[1]][end_pos[0]] < min_val) \
                or (max_val is not None and inp_arrs[0][end_pos[1]][end_pos[0]] > max_val) \
                or (traversability_expression is not None and not eval_expression_at_pos(traversability_expression_str,
                                                                                         end_pos, inp_arrs,
                                                                                         traversability_expression)):
            raise ValueError(self.tr("Ending Point must be in a traversable area"))

        # A* Algorithm
        frontier = PriorityQueue()
        frontier.put(start_pos, 0)

        came_from = np.empty(inp_arrs[0].shape, np.ushort)
        came_from[start_pos[1]][start_pos[0]] = 0

        cost_so_far = np.full(inp_arrs[0].shape, np.inf, np.float64)
        cost_so_far[start_pos[1]][start_pos[0]] = 0

        starting_heuristic = a_star_heuristic(start_pos, end_pos)
        min_heuristic = starting_heuristic

        warned_inadmissible_heuristic = False

        print("Starting A*")
        while True:
            if feedback.isCanceled():
                raise RuntimeError("Task Cancelled")
            feedback.setProgress((1 - min_heuristic / starting_heuristic) * 100)

            if frontier.empty():
                raise ValueError(self.tr("No path found"))
            current = frontier.get()

            if current == end_pos:
                break

            neighbors = [n for n in get_neighbors(current, inp_arrs[0].shape[1], inp_arrs[0].shape[1]) if
                         (min_val is None or inp_arrs[0][n[0][1]][n[0][0]] >= min_val) and
                         (max_val is None or inp_arrs[0][n[0][1]][n[0][0]] <= max_val) and
                         (traversability_expression is None or
                          eval_expression_at_pos(traversability_expression_str, n, inp_arrs, traversability_expression))
                         ]
            for next_pos, direction in neighbors:
                if cost_expression is None:
                    add_cost = 1
                else:
                    add_cost = eval_expression_at_pos(cost_expression_str, next_pos, inp_arrs, cost_expression)
                    if not warned_inadmissible_heuristic and add_cost < 1:
                        warned_inadmissible_heuristic = True
                        feedback.pushInfo(
                            self.tr("[WARNING] Custom cost expression is less than 1, path may not be optimal!"))

                new_cost = cost_so_far[current[1]][current[0]] + add_cost
                if new_cost < cost_so_far[next_pos[1]][next_pos[0]]:
                    cost_so_far[next_pos[1]][next_pos[0]] = new_cost
                    heuristic = a_star_heuristic(next_pos, end_pos)
                    min_heuristic = min(min_heuristic, heuristic)
                    frontier.put(next_pos, new_cost + heuristic)
                    came_from[next_pos[1]][next_pos[0]] = DIRECTION_MAPPING[direction]

        print("Reconstructing path")
        current_point = end_pos
        last_point = None
        path = []
        while True:
            print(current_point)
            if feedback.isCanceled():
                raise RuntimeError("Task Cancelled")
            delta_code = came_from[current_point[1]][current_point[0]]
            if delta_code == 0:
                next_point = None
            else:
                delta = DIRECTION_MAPPING_INV[delta_code]
                next_point = (current_point[0] - delta[0], current_point[1] - delta[1])
            if last_point is None or next_point is None or \
                    next_point[0] - current_point[0] != current_point[0] - last_point[0] or \
                    next_point[1] - current_point[1] != current_point[1] - last_point[1]:
                path.append(pixel_to_point(current_point, bounding_rect,
                                           inp_arrs[0].shape[1], inp_arrs[0].shape[0]))

            if current_point == start_pos:
                break
            last_point = current_point
            current_point = next_point

        # Add a feature in the sink
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPolyline(path))
        sink.addFeature(feature)

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: dest_id}

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Find Path'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return ''

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return PathfinderAlgorithm()
