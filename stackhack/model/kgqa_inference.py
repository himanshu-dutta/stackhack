from transformers import AutoTokenizer, T5ForConditionalGeneration
from stackhack.data.preprocessing.kgqa import KGQA_Item
import argparse
import json


class InferenceEngine:
    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path
        )

    def preprocess(self, ques: str, kg: dict) -> str:
        itm = KGQA_Item(ques, kg)
        itm = itm.gen_datapoint()
        return itm["input"]

    def infer(self, ques: str, kg: dict, max_target_length: int) -> str:
        input_ids = self.tokenizer(
            self.preprocess(ques, kg), return_tensors="pt", max_length=max_target_length
        ).input_ids

        outputs = self.model.generate(input_ids, max_length=max_target_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--ques", type=str, required=True)
    parser.add_argument("-k", "--kg_path", type=str, required=True)
    parser.add_argument("-m", "--model_save_path", type=str, required=True)
    parser.add_argument(
        "-t",
        "--max_target_length",
        type=int,
        choices=[256, 512, 1024, 2048],
        default=512,
    )
    args = parser.parse_args()

    ques = args.ques
    kg_path = args.kg_path
    model_save_path = args.model_save_path
    max_target_length = args.max_target_length

    with open(kg_path, "r") as fp:
        kg = json.load(fp)

    ie = InferenceEngine(model_save_path)
    ans = ie.infer(ques, kg, max_target_length)

    print(f"QUESTION: {ques}")
    print("=================================================================")
    print(f"ANSWER: {ans}")
