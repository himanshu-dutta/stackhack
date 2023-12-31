import datasets
import argparse
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os
import random

MODEL_CHECKPOINT_OPTIONS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]


def load_dataset(
    dataset_path: str, data_frac: float, train_pct: float, valid_pct: float
):
    assert (
        train_pct + valid_pct < 1.0
    ), "Can't have more than 100% of the data in the splits"

    data_perc = int(data_frac * 100)
    all_files = os.listdir(dataset_path)
    num_files_to_load = int(len(all_files) * (data_perc / 100))
    selected_files = random.sample(all_files, num_files_to_load)
    data_files = [os.path.join(dataset_path, file) for file in selected_files]
    dataset = datasets.load_dataset("json", data_files=data_files)

    valid_pct = valid_pct / (1 - train_pct)
    train_testval_ds = dataset["train"].train_test_split(test_size=1.0 - train_pct)
    test_valid_ds = train_testval_ds["test"].train_test_split(test_size=valid_pct)

    train_test_valid_dataset = datasets.DatasetDict(
        {
            "train": train_testval_ds["train"],
            "test": test_valid_ds["test"],
            "validation": test_valid_ds["train"],
        }
    )

    return train_test_valid_dataset


def make_preprocess_function(tokenizer, max_target_length=512):
    def preprocess_function(examples):
        inputs = [doc for doc in examples["input"]]

        model_inputs = tokenizer(
            inputs,
            max_length=max_target_length,
            # max_length=max([len(inp) for inp in inputs]),
            return_tensors="pt",
            padding="max_length",
            truncation=True
            # truncation=False
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=examples["output"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True)
    parser.add_argument("-f", "--data_frac", type=float, default=1.0)
    parser.add_argument(
        "-c",
        "--checkpoint",
        choices=MODEL_CHECKPOINT_OPTIONS,
        default=MODEL_CHECKPOINT_OPTIONS[0],
    )
    parser.add_argument(
        "-t",
        "--max_target_length",
        type=int,
        choices=[256, 512, 1024, 2048],
        default=512,
    )
    parser.add_argument("-s", "--model_save_path", type=str, required=True)
    parser.add_argument("-e", "--num_epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)

    args = parser.parse_args()

    data_dir = args.data_dir
    data_frac = args.data_frac
    model_checkpoint = args.checkpoint
    max_target_length = args.max_target_length
    model_save_path = args.model_save_path

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    raw_datasets = load_dataset(data_dir, data_frac, 0.7, 0.2)
    tokenized_datasets = raw_datasets.map(
        make_preprocess_function(tokenizer, max_target_length), batched=True
    )
    print("Loaded the dataset: ", tokenized_datasets)

    batch_size = args.batch_size
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-stackhack",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(model_save_path)
