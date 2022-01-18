import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProject,
                       QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterPoint,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterString,
                       QgsProcessingParameterFeatureSink,
                       QgsPoint,
                       QgsRectangle,
                       QgsWkbTypes,
                       QgsFields)
import typing as t
import heapq
from osgeo import gdal

from .formulas import parse_formula, evaluate_formula


class PriorityQueue:
    def __init__(self):
        self.elements: t.List[t.Tuple[float, (int, int)]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: (int, int), priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> (int, int):
        return heapq.heappop(self.elements)[1]

    def __len__(self):
        return len(self.elements)


class PathfinderAlgorithm(QgsProcessingAlgorithm):
    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.
    OUTPUT = 'OUTPUT'

    INPUT_IMAGES = ["INPUT_IMG" + str(i) for i in range(3)]
    INPUT_POINT1 = 'INPUT_POINT1'
    INPUT_POINT2 = 'INPUT_POINT2'
    INPUT_MIN_VAL = 'INPUT_MIN_VAL'
    INPUT_MAX_VAL = 'INPUT_MAX_VAL'
    INPUT_TRAVERSABILITY_ENUM = 'INPUT_TRAVERSABILITY_ENUM'
    INPUT_COST_ENUM = 'INPUT_COST_ENUM'
    INPUT_TRAVERSABILITY_EXPRESSION = 'INPUT_TRAVERSABILITY_EXPRESSION'
    INPUT_COST_EXPRESSION = 'INPUT_COST_EXPRESSION'

    def __init__(self):
        super().__init__()

        self.inp_arrs: t.List[np.ndarray] = []
        self.bounding_rect: QgsRectangle = None
        self.grid_width: t.Optional[int] = None
        self.grid_height: t.Optional[int] = None
        self.basis_arr: t.Optional[np.ndarray] = None

        self.start_point: QgsPoint = None
        self.end_point: QgsPoint = None

        self.is_traversable = None
        self.get_cost = None

        self.output_id = None
        self.output_sink = None

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
            QgsProcessingParameterEnum(
                self.INPUT_TRAVERSABILITY_ENUM,
                self.tr("Traversability Layer"),
                (
                    "Primary",
                    "Secondary",
                    "Tertiary",
                    "Custom/None"
                ),
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.INPUT_MIN_VAL,
                self.tr('Minimum Traversable Value'),
                defaultValue=1
            ),
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
            QgsProcessingParameterEnum(
                self.INPUT_COST_ENUM,
                self.tr("Cost Layer"),
                (
                    "Primary",
                    "Secondary",
                    "Tertiary",
                    "Custom/None"
                ),
                defaultValue=1
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

    def get_expression_vars(self, pos: (int, int)) -> t.Dict[str, t.Any]:
        vars_dict = {}

        x = pos[0]
        y = pos[1]
        vars_dict["x"] = x
        vars_dict["y"] = y

        for i, arr in enumerate(self.inp_arrs):
            if arr is not None:
                s = str(i + 1)
                vars_dict["val" + s] = arr[y][x]

        return vars_dict

    def parse_inputs(self, parameters, context):
        inp_layers = []
        basis_layer = None
        for i, code in enumerate(self.INPUT_IMAGES):
            param = self.parameterAsRasterLayer(parameters, code, context)
            inp_layers.append(param)
            if basis_layer is None:
                basis_layer = i
        if basis_layer is None:
            raise ValueError(self.tr("At least one raster input is required"))

        self.inp_arrs = []
        for layer in inp_layers:
            if layer is not None:
                img_ds = gdal.Open(layer.dataProvider().dataSourceUri())
                self.inp_arrs.append(img_ds.GetRasterBand(1).ReadAsArray())
            else:
                self.inp_arrs.append(None)

        self.start_point = self.parameterAsPoint(parameters, self.INPUT_POINT1, context)
        self.end_point = self.parameterAsPoint(parameters, self.INPUT_POINT2, context)

        traversability_enum = self.parameterAsEnum(parameters, self.INPUT_TRAVERSABILITY_ENUM, context)
        traversability_layer = self.inp_arrs[traversability_enum] \
            if traversability_enum < len(self.INPUT_IMAGES) else None

        if traversability_layer is not None:
            traversability_min = self.parameterAsString(parameters, self.INPUT_MIN_VAL, context)
            try:
                traversability_min = float(traversability_min)
            except ValueError:
                traversability_min = None
            transversability_max = self.parameterAsString(parameters, self.INPUT_MAX_VAL, context)
            try:
                traversability_max = float(transversability_max)
            except ValueError:
                traversability_max = None

            if traversability_min is None and traversability_max is None:
                self.is_traversable = lambda pos: True
            elif traversability_max is not None:
                self.is_traversable = lambda pos: traversability_layer[pos[1]][pos[0]] <= traversability_max
            elif traversability_min is not None:
                self.is_traversable = lambda pos: traversability_layer[pos[1]][pos[0]] >= traversability_min
            else:
                self.is_traversable = lambda pos: \
                    traversability_min <= traversability_layer[pos[1]][pos[0]] <= traversability_max

        else:
            traversability_expression_str = self.parameterAsString(
                parameters, self.INPUT_TRAVERSABILITY_EXPRESSION, context)
            if traversability_expression_str == "":
                self.is_traversable = lambda pos: True
            else:
                traversability_expression = parse_formula(traversability_expression_str)
                self.is_traversable = lambda pos: evaluate_formula(
                    traversability_expression_str, self.get_expression_vars(pos), traversability_expression
                )

        cost_enum = self.parameterAsEnum(parameters, self.INPUT_COST_ENUM, context)
        cost_layer = self.inp_arrs[cost_enum] if cost_enum < len(self.INPUT_IMAGES) else None

        if cost_layer is not None:
            self.get_cost = lambda pos: cost_layer[pos[1]][pos[0]]
        else:
            cost_expression_str = self.parameterAsString(parameters, self.INPUT_COST_EXPRESSION, context)
            if cost_expression_str == "":
                self.get_cost = lambda pos: 1
            else:
                cost_expression = parse_formula(cost_expression_str)
                self.get_cost = lambda pos: evaluate_formula(
                    cost_expression_str, self.get_expression_vars(pos), cost_expression
                )

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, QgsFields(),
                                             geometryType=QgsWkbTypes.Type.LineString,
                                             crs=inp_layers[basis_layer].crs())
        self.output_sink = sink
        self.output_id = dest_id

        self.bounding_rect = inp_layers[basis_layer].extent()
        self.grid_width = self.inp_arrs[basis_layer].shape[1]
        self.grid_height = self.inp_arrs[basis_layer].shape[0]

        if not self.bounding_rect.contains(self.start_point):
            raise ValueError(self.tr("Starting Point must be somewhere within the first raster image"))
        if not self.bounding_rect.contains(self.end_point):
            raise ValueError(self.tr("Ending Point must be somewhere within the first raster image"))

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
