# From https://stackoverflow.com/a/287944/16569836
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import os

MODEL_PATH_LWM = os.environ.get("LWM_WEIGHT_PATH", None)
MODEL_PATH_LWM_DISTSERVE = os.environ.get("LWM_WEIGHT_DISTSERVE_PATH", None)
EXP_RESULT_PATH = os.environ.get("EXP_RESULT_ROOT_PATH")

assert MODEL_PATH_LWM is not None, "Env `LWM_WEIGHT_PATH` is not set!"
assert MODEL_PATH_LWM_DISTSERVE is not None, "Env `LWM_WEIGHT_DISTSERVE_PATH` is not set!"
assert EXP_RESULT_PATH is not None, "Env `EXP_RESULT_ROOT_PATH` is not set!"

LOONGSERVE_DB_IDENTICAL_REQ_PATH = str(os.path.join(EXP_RESULT_PATH, "loongserve-db-identical-req.sqlite"))
VLLM_DB_IDENTICAL_REQ_PATH = str(os.path.join(EXP_RESULT_PATH, "vllm-db-identical-req.sqlite"))
