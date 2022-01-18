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

import math
import typing as t
import numpy as np
from qgis.core import (QgsProject,
                       QgsProcessing,
                       QgsWkbTypes,
                       QgsFields,
                       QgsFeature,
                       QgsGeometry,
                       QgsPoint,
                       QgsRectangle)
from .pathfinder_algorithm import PathfinderAlgorithm, PriorityQueue

NEIGHBORS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def theta_star_heuristic(pos1: (int, int), pos2: (int, int)) -> float:
    return math.dist(pos1, pos2)


class AnyAnglePathfinderAlgorithm(PathfinderAlgorithm):
    """
    This algorithm uses the Theta* algorithm to find near-optimal paths within raster images. The user may specify
    custom expressions cost and traversability of the image. The output paths may have segments at any angle.
    """

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        self.parse_inputs(parameters, context)

        start_pos = point_to_pixel(self.start_point, self.bounding_rect, self.grid_width, self.grid_height)
        if not self.is_traversable(start_pos):
            raise ValueError(self.tr("Starting point must be traversable"))
        end_pos = point_to_pixel(self.end_point, self.bounding_rect, self.grid_width, self.grid_height)
        if not self.is_traversable(end_pos):
            raise ValueError(self.tr("Ending point must be traversable"))

        # Theta* Algorithm
        frontier = PriorityQueue()
        frontier.put(start_pos, 0)

        came_from = np.empty((self.grid_height, self.grid_width), np.ushort)
        came_from[start_pos[1]][start_pos[0]] = 0

        cost_so_far = np.full((self.grid_height, self.grid_width), np.inf, np.float64)
        cost_so_far[start_pos[1]][start_pos[0]] = 0

        starting_heuristic = theta_star_heuristic(start_pos, end_pos)
        min_heuristic = starting_heuristic

        warned_inadmissible_heuristic = False

        print("Starting Theta*")
        while True:
            if feedback.isCanceled():
                raise RuntimeError("Task Cancelled")
            feedback.setProgress((1 - min_heuristic / starting_heuristic) * 100)

            if frontier.empty():
                raise ValueError(self.tr("No path found"))
            current = frontier.get()

            if len(frontier) > 100000:
                print(frontier.elements)
                raise RuntimeError(self.tr("Max queue length exceeded"))

            if current == end_pos:
                break

            neighbors = [n for n in get_neighbors(current, self.grid_width, self.grid_height) if
                         self.is_traversable(n[0])]
            for next_pos, direction in neighbors:
                add_cost = self.get_cost(next_pos)
                if not warned_inadmissible_heuristic and add_cost < 1:
                    warned_inadmissible_heuristic = True
                    feedback.pushInfo(
                        self.tr("[WARNING] Custom cost expression is less than 1, path may not be optimal!"))

                new_cost = cost_so_far[current[1]][current[0]] + add_cost
                if new_cost < cost_so_far[next_pos[1]][next_pos[0]]:
                    cost_so_far[next_pos[1]][next_pos[0]] = new_cost
                    heuristic = theta_star_heuristic(next_pos, end_pos)
                    min_heuristic = min(min_heuristic, heuristic)
                    frontier.put(next_pos, new_cost + heuristic)
                    came_from[next_pos[1]][next_pos[0]] = DIRECTION_MAPPING[direction]

        print("Reconstructing path")
        current_point = end_pos
        last_point = None
        path = []
        while True:
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
                path.append(pixel_to_point(current_point, self.bounding_rect, self.grid_width, self.grid_height))

            if current_point == start_pos:
                break
            last_point = current_point
            current_point = next_point

        # Add a feature in the sink
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPolyline(path))
        self.output_sink.addFeature(feature)

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: self.output_id}

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'Find Path (Any Angle)'

    def createInstance(self):
        return AnyAnglePathfinderAlgorithm()