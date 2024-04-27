import os
import numpy as np

# PARAMETERS YOU DONT HAVE TO CHANGE
Z_PERCENTILE = 0.5  # it means 0.15% lower points are deleted (errors, artifacts, garbage)
FIGSIZE = 30
FIGSIZE_HEIGHT_MAP_GREY = 30
DPI_HEIGHT_MAP_GREY = 300
MARKER = 'o'
MARKER_SIZE = 0.01
INDEX_TO_PARSE = 0
LOCAL_HEIGHTS_RATIO_CROP = 0.2


# PARAMETERS YOU MAY WANT TO CHANGE:
TEST = False  # MUST BE "FALSE" FOR PRODUCTION. if TEST is True - no data is parsed, generates random XYZ points.
TEST_SAMPLE_SIZE = 100  # number of random XYZ points for test
TREE_FILENAME = 'moesc/df_tree.csv'
MIN_HEIGHT = 1.5  # min height of trees in output data
BOTTOM_PERCENTILE = 10  # it means bottom 10%. Used in defining local bottom of the ground

# PARAMETERS YOU PROBABLY DONT WANT TO CHANGE
# IF YOU WANT TO CHANGE BASE_DPI, ALSO CHANGE ALL PARAMETERS BELOW EQUIVALENTIALLY.
# E.G.: if BASE_DPI is changed to 100 (divided by 2), maximum_filter_size has to be 7 (divided by 2)
BASE_DPI = 200  # If memory consumption is too high, decrease to 100-150
MAXIMUM_FILTER_SIZE = 15  # increased
ERODE_VALUE = 5
NEIGHBOUR_RADIUS = 200
NEIGHBOUR_RADIUS_LOCAL_HEIGHTS = 50
MIN_LABEL_SIZE = 2
MAX_LABEL_SIZE = 350
LABEL_EXCLUDING_MARGIN = 50

#google map
DIST = 5
TREE_COLOR_FILL = '#00FF00'
TREE_COLOR_STROKE = '#008000'
TREE_FILL_OPACITY = 1
ZOOM = 18

POWER_LINE_OPACITY = 0.5
TEST = False  # must be False for production

#PIL Image
MAX_IMAGE_PIXELS = 2000000000

#keras model
RESIZE = True
LOW_MEMORY = True

# paths
FILEPATH_IN_LINES = 'moesc/MOESC_lines.csv'
QUEUE_PATH = os.path.join('moesc', 'queue.json')
LOG_PATH = ('moesc.log', 'log')
MODEL_PATHS = [os.path.join('moesc', 'model', 'all.h5'),
               os.path.join('moesc', 'model',
                            'model_p_try2_0_0.8172824813728132.pth')]
KML_DIR = 'KML_FILES'

# NN SEGMENTATION
# TREE GREEN
NN_DEVICE = 'cpu'
MODEL_PATH_GREEN_TREE = 'moesc/model/tree_green.pth'
ENCODER_GREEN_TREE = 'timm-efficientnet-b3'
ENCODER_WEIGHTS_GREEN_TREE = 'imagenet'
TILE_SIZE_GREEN_TREE = 320

# TREE DRY
MODEL_PATH_DRY_TREE = 'moesc/model/tree_dry.pth'
ENCODER_DRY_TREE = 'timm-efficientnet-b3'
ENCODER_WEIGHTS_DRY_TREE = 'imagenet'
TILE_SIZE_DRY_TREE = 320

# WIRES DETECTION
MODEL_PATH_DRY_TREE = 'moesc/model/wires_model.pth'
ENCODER_DRY_TREE = 'timm-efficientnet-b3'
ENCODER_WEIGHTS_DRY_TREE = 'imagenet'
TILE_SIZE_DRY_TREE = 320

# SPLIT TO SPANS
SPLIT_PROCESSES_COUNT = 17
DTYPE = np.uint16
SPLIT_RESIZE_COEF = 4
INCREMENT_PRECISION = 0.01
MAX_DISTANCE = 25  # in meters

# SPANS
RADIUS_PILLAR_INSIDE_METERS = 15  # in meters
RADIUS_PILLAR_INSIDE_BLACK_RATIO_THRESHOLD = 0.25

# COLORS
SUM_MASK_GREEN = np.array([0, 0, 255]).astype('uint8')  # red
SUM_MASK_DRY = np.array([255, 0, 180]).astype('uint8')  # red

# DRAWING
PILLAR_RADIUS_MAJOR = 130
PILLAR_WIDTH_MAJOR = 25
PILLAR_COLOR = [255, 0, 0]  # [0, 0, 255] red in BGR


PILLAR_RADIUS = 15
PILLAR_WIDTH = 5

POWERLINE_COLOR = [255, 0, 0]  # blue?

BACKGROUND_WEIGHT = 0.4  # used in draw to make transparency


TREE_RADIUS = 25
TREE_WIDTH = 10
COLOR_TREE_GREEN_IN_ZONE = [0, 0, 255]  # red without bgr2rgb
COLOR_TREE_GREEN_OUT_ZONE = [0, 255, 0]  # green without bgr2rgb

COLOR_TREE_DRY_IN_ZONE = [255, 0, 180]  # red without bgr2rgb
COLOR_TREE_DRY_OUT_ZONE = [255, 0, 255]  # green without bgr2rgb

POWERLINE_WIDTH = 5  # line thickness
INCLOSURE_COLOR = [139, 0, 0]
INCLOSURE_RADIUS = 10
INCLOSURE_LINE_WIDTH = 7  # line thickness
INCLOSURE_WIDTH = 5

# FILTER TREES
SECURE_DISTANCE = 5
LINE_BINS = 100  # points within line to calc if tree is in zone

# FILENAMES
FILENAME_POINTCLOUD = 'pointcloud.txt'
FILENAME_POINTCLOUD_MAJOR = 'odm_georeferenced_model.txt'
FILENAME_ORTHOPHOTO_MAJOR_TIFF = 'odm_orthophoto.tif'
FILENAME_ORTHOPHOTO_MAJOR_JPG = 'odm_orthophoto.jpg'
FILENAME_ORTHOPHOTO_MAJOR = 'odm_orthophoto.tif'
FILENAME_ORTHOPHOTO_MAJOR_ROTATED = 'odm_orthophoto_rotated.jpg'
FILENAME_ORTHOPHOTO_PILLARS_MAJOR = 'orthophoto_with_pillars.jpg'

FILENAME_MASK_GREEN = 'mask_green_tree.png'
FILENAME_MASK_DRY = 'mask_dry_tree.png'
WIRES_MASK = 'mask_wires.png'

FILENAME_MASK_GREEN_ED = 'mask_green_tree_ED.png'
FILENAME_MASK_DRY_ED = 'mask_dry_tree_ED.png'

FILENAME_ORTHOPHOTO = 'orthophoto.jpg'
FILENAME_ORTHOPHOTO_WITH_MASK = 'orthophoto_with_mask.jpg'
FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS = 'orthophoto_with_mask_pillars.jpg'

FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE = 'orthophoto_with_mask_pillars_powerline.jpg'
FILENAME_ORTHOPHOTO_WITH_MASK_PILLARS_POWERLINE_MAXIMA = 'orthophoto_with_mask_pillars_powerline_maxima.jpg'

FILENAME_LOCAL_MAXIMA_CSV = 'local_maxima.csv'
FILENAME_LOCAL_MAXIMA_MAJOR_CSV = 'local_maxima.csv'
FILENAME_LOCAL_MAXIMA_SUPPLEMENT_CSV = 'local_maxima_supplement.csv'
FILENAME_LOCAL_MAXIMA_PNG = 'local_maxima.png'

FILENAME_SPANS = 'spans.csv'
FILENAME_SPANS_ALL = 'spans_all.csv'

FILENAME_IMAGE_AFTER_SPLIT = 'df_image_after_split.pickle'
FILENAME_SPLITTED_SCHEME = 'scheme_splitted.png'

FILENAME_INCLOSURE_LINES = 'inclosure_lines.csv'
FILENAME_INCLOSURE_MASK = 'inclosure_mask.png'

FILENAME_TREE_AREA_MINOR = 'tree_area.csv'
FILENAME_TREE_AREA_MINOR_SUPPLEMENT = 'tree_area_supplement.csv'
FILENAME_TREE_AREA_MAJOR = 'tree_area.csv'

FILENAME_RESULT_JSON = 'result.json'
FILENAME_RESULT_XLSX = 'result.xlsx'

FILENAME_ZIP_ORTHOPHOTOS = 'orthophotos.zip'

FILENAME_ORTHOPHOTO_ALIGNED = 'orthophoto_aligned.jpg'
FILENAME_ORTHOPHOTO_ALIGNED_CROPPED = 'orthophoto_aligned_cropped.jpg'
FILENAME_SCHEME_SPANS = 'scheme_spans.png'
