import json
import argparse
from tqdm import tqdm
from pathlib import Path
import os
from bs4 import BeautifulSoup

# the available data
# question, answer, relevant kg


def parse_html(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text()


class KGQA_Item:
    def __init__(self, ques, kg, ans):
        self.ques = ques
        self.kg = kg
        self.ans = ans

    def _stringify_kg(self):
        st = ""
        for rel in self.kg["relations"]:
            st += f'{rel["head"]} {rel["type"]} {rel["tail"]}. '
        return st

    def gen_datapoint(self):
        return {
            "input": f"QUESTION: {self.ques}, CONTEXT: {self._stringify_kg()}",
            "output": self.ans,
        }


def preprocess(doc: dict, kg: dict, threshold: int):
    # rank the answers to a question based on the upvotes
    # decide a threshold, k, and select the top k answers for the dataset
    ques = doc.get("body", None)
    answers = doc.get("answers", None)

    if answers is None:
        return dict()

    answers = sorted(answers, key=lambda ans: int(ans["up_vote_count"]), reverse=True)
    answers = answers[:threshold]

    datapoints = {
        f'{doc["question_id"]}_{ans["answer_id"]}': KGQA_Item(
            parse_html(ques), kg, parse_html(ans["body"])
        ).gen_datapoint()
        for ans in answers
        if int(ans["up_vote_count"]) > 0
    }
    return datapoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--docs_dir_path", type=str, required=True)
    parser.add_argument("-k", "--kgs_dir_path", type=str, required=True)
    parser.add_argument("-s", "--save_dir_path", type=str, required=True)
    parser.add_argument("-t", "--threshold", type=int, default=3)
    args = parser.parse_args()

    docs_dir_path = Path(args.docs_dir_path)
    save_dir_path = Path(args.save_dir_path)
    kg_dir_path = Path(args.kgs_dir_path)
    threshold = args.threshold

    doc_files = [
        file for file in os.listdir(docs_dir_path.absolute()) if file.endswith(".json")
    ]

    print("Number of files: ", len(doc_files))
    total_datapoint_count = 0

    for doc_file_name in tqdm(doc_files):
        doc_file_path = docs_dir_path / doc_file_name
        kg_file_path = kg_dir_path / doc_file_name

        if not (os.path.isfile(doc_file_path) and os.path.isfile(kg_file_path)):
            continue

        with open(doc_file_path, "r") as fp:
            doc = json.load(fp)
        with open(kg_file_path, "r") as fp:
            kg = json.load(fp)

        datapoints = preprocess(doc, kg, threshold)
        total_datapoint_count += len(datapoints)
        for id, dp in datapoints.items():
            save_path = save_dir_path / (id + ".json")
            with open(save_path, "w") as fp:
                json.dump(dp, fp)

    print(f"Number of datapoints generated: {total_datapoint_count}")
