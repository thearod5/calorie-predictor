import json
import os

import cv2

from constants import DEFAULT_TURK_RESULTS
from src.datasets.menu_match_dataset import MenuMatchDataset
from src.turk_task.hmap_creator import HMapCreator

DATASET = MenuMatchDataset()
INPUT_DIR = DATASET.DATASET_PATH_CREATOR.image_dir
DEMO_DIR = "~/desktop/cpred"


def export(image, path: str):
    image = cv2.resize(image, (250, 250))
    cv2.imwrite(path, image)
    print("Exported:", path)


if __name__ == "__main__":
    # A. Constants
    image_name = "img5.jpg"
    image_input_path = os.path.join(INPUT_DIR, image_name)
    image_export_path = os.path.join(DEMO_DIR, image_name)
    results_path = os.path.expanduser(os.path.join(DEFAULT_TURK_RESULTS, "results.csv"))
    original_export_path = os.path.expanduser(os.path.join(DEMO_DIR, "source.jpg"))
    bounding_box_export_path = os.path.expanduser(os.path.join(DEMO_DIR, "bounding_box.jpg"))
    hmap_export_path = os.path.expanduser(os.path.join(DEMO_DIR, "hmap.jpg"))

    # 1. Save original image to demo folder
    hmap_creator = HMapCreator(DATASET, ["results.csv"])
    img = cv2.imread(image_input_path)
    result = hmap_creator.get_image_result(image_name)
    export(img, original_export_path)

    # 2. Save image with bounding box
    bounding_boxes = json.loads(result["Answer.annotatedResult.boundingBoxes"])[0]
    (start_x, start_y, end_x, end_y) = HMapCreator.get_bounding_box_coordinates(bounding_boxes)
    print((start_x, start_y, end_x, end_y))
    bounding_box_img = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 10)
    export(bounding_box_img, bounding_box_export_path)

    # 3. Save heatmap with blur
    hmap_img = hmap_creator.create_hmap_for_image(result, DEMO_DIR, write_image=False)
    hmap_img = cv2.blur(hmap_img, ksize=(250, 250))
    export(hmap_img, hmap_export_path)
    print("Done!")
