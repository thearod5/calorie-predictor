import json
import os
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd

RUN_NAME = "batch-1"
FILE_NAME = "results.csv"
PATH_TO_RESULTS = os.path.join("~", "desktop", "data", "calorie-predictor", "turk-task", "results")
PATH_TO_RESULTS_DATA = os.path.join(PATH_TO_RESULTS, RUN_NAME, FILE_NAME)
RESULT_INDEX = 0
COLOR = (255, 0, 0)


def get_image_coordinates(image_box: dict):
    start_x = image_box["left"]
    start_y = image_box["top"]
    end_x = start_x + image_box["width"]
    end_y = start_y + image_box["height"]
    return start_x, start_y, end_x, end_y


def read_image(image_url):
    req = urlopen(image_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return cv2.imdecode(arr, -1)  # 'Load it as it is'


def display_result(result):
    """
    box coordinates returned from your model's predictions
    color is the color of the bounding box you would like & 2 is the thickness of the bounding box
    :param result:
    :type result:
    :return:
    :rtype:
    """
    result_image_url = result["Input.image_url"]
    bounding_boxes = json.loads(result["Answer.annotatedResult.boundingBoxes"])
    if len(bounding_boxes) == 0:
        print("Missing bounding box: " + result["HITId"])
        return
    result_image_box = bounding_boxes[0]
    input_image = read_image(result_image_url)

    file_name = result_image_url.split("/")[-1]
    (start_x, start_y, end_x, end_y) = get_image_coordinates(result_image_box)
    image = cv2.rectangle(input_image, (start_x, start_y), (end_x, end_y), COLOR)

    output_image_path = os.path.join(PATH_TO_RESULTS, RUN_NAME, file_name)
    cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    results_df = pd.read_csv(PATH_TO_RESULTS_DATA)
    for i in range(len(results_df)):
        display_result(results_df.iloc[i])
    print("Done!")
