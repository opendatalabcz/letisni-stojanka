CLASS_NAMES = {"0": "airplane",\
                "1": "cargo-door", \
               "2": "jet-bridge", \
               "3": "tank-truck", \
               "4": "cargo-truck", \
               "5": "push-back-truck", \
               "6": "luggage-loading-truck", \
               "7": "cargo-box", \
               "8": "basic-truck", \
               "9": "passenger-bus", \
               "10": "person",}

DATA_PATH = "/home/oliver/School/THESIS/data/dataset"

TOTAL_FRAMES = 320427

VALID_START = 104422
VALID_STOP = 131422

TEST_START = 50423
TEST_STOP = 77423

NORMAL_FR = 30
IMP_FR = 10
TEST_FR = 10

IMPORTANT_SCENES = {"111809": "113722",
                    "119922": "124422",
                    "126422": "127272",
                    "137757": "145681",
                    "147961": "148571",
                    "156822": "157269",
                    "158792": "159652",
                    "161642": "162252",
                    "163382": "164402",
                    "167292": "169032",
                    "173112": "176422",
                    "181672": "183622",
                    "195632": "199823",
                    "201122": "202422",
                    "204422": "206722",
                    "226772": "230422",
                    "233422": "235422",
                    "255422": "258422",
                    "284422": "287922", }


NOT_SCENES = {"104422": "107422",
            "127422": "130422",
            "185432": "190889",
            "212472": "216822",
            "237272": "239425",
            "239426": "242252",
            "243422": "250422",
            "264422": "284422",}
                    
TRAIN_FILE = "/home/blaskoli/cfg/train.txt"
TEST_FILE = "/home/blaskoli/cfg/test.txt"
VALID_FILE = "/home/blaskoli/cfg/valid.txt"

IMG_W = 1280
IMG_H = 720

YOLOV4_CONFIG_PATH = "/home/oliver/School/THESIS/letisni-stojanka/model_yolov4/yolo-obj.cfg"
YOLOV4_WEIGHTS_PATH = "/home/oliver/School/THESIS/letisni-stojanka/model_yolov4/yolo-obj_best.weights"

TINY_CONFIG_PATH = "/home/oliver/School/THESIS/letisni-stojanka/model_tiny/yolov4-tiny-custom.cfg"
TINY_WEIGHTS_PATH = "/home/oliver/School/THESIS/letisni-stojanka/model_tiny/yolov4-tiny-custom_best.weights .weights"

LABELS_PATH = "/home/oliver/School/THESIS/letisni-stojanka/model_yolov4/obj.names"

OUT_PATH = "/home/oliver/School/THESIS/letisni-stojanka/front_end/my-app/src/video/"