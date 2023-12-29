import json
import argparse
from tqdm import tqdm
from pathlib import Path


def preprocess(doc: dict):
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-s", "--save_dir_path", type=str, required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    save_dir_path = Path(args.save_dir_path)

    with open(data_path, "r") as fp:
        raw_data = json.load(fp)

    for doc in tqdm(raw_data):
        save_path = save_dir_path / f'{doc["question_id"]}.json'
        doc = preprocess(doc)
        with open(save_path, "w") as fp:
            json.dump(doc, fp)
