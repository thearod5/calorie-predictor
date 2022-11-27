from src.datasets.menu_match_dataset import MenuMatchDataset
from src.turk_task.hmap_creator import HMapCreator

if __name__ == "__main__":
    MODE = "save_avg_hmap"
    creator = HMapCreator(MenuMatchDataset(), ["results.csv"])
    if MODE == "save_batches":
        creator.save_hmap_batches()
    if MODE == "save_avg_hmap":
        creator.save_avg_hmaps()
