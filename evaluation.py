import os
import pickle
import zlib

import datasets
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainingArguments
from datasets import Audio
import evaluate
from peft import prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import torch
import shutil
import subprocess
from langdetect import detect
import subprocess
from huggingface_hub import scan_cache_dir
from sklearn.model_selection import train_test_split

os.environ["HF_DATASETS_CACHE"] = "None"

login(token = "HF_TOKEN",add_to_git_credential=True)

model_name_or_path = "openai/whisper-large-v3"

#model_name_or_path="devasheeshG/whisper_large_v2_fp16_transformers"

#model_name_or_path="airesearch/wav2vec2-large-xlsr-53-th"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_17_0"

law_data = datasets.load_from_disk("med-dataset")


from datasets import Dataset, concatenate_datasets

# Example combined dataset (replace this with your actual dataset)
# combined_dataset = ...  # Load your actual dataset here

# Sentence counts for each language
sentence_counts = [1288, 416, 1843, 33, 3739, 3602, 1583, 2227, 58, 116, 177, 2574, 187, 13, 919, 276, 637, 130, 679, 666,556, 69, 516]


new_datasets_train = []
new_datasets_test = []
new_datasets_eval = []

start_index = 0
for count in sentence_counts:
    end_index = start_index + count
    last_20_percent_index = int(count * 0.8)  # Get the index for the last 20%

    # Calculate indices for splits
    train_end_index = int(count * 0.8)  # First 80% for training
    test_end_index = int(count * 0.9)  # Next 10% for testing
    eval_end_index = count  # Remaining 10% for evaluation

    # Extract datasets for current language
    if count != 1:
        train_language_subset = law_data.select(range(start_index, start_index + train_end_index))
        test_language_subset = law_data.select(
            range(start_index + int(count * 0.8), start_index + int(count * 0.9)))
        eval_language_subset = law_data.select(range(start_index + int(count * 0.9), start_index + count))
        print(test_language_subset)
        # Append to respective datasets
        new_datasets_train.append(train_language_subset)
        new_datasets_test.append(test_language_subset)
        new_datasets_eval.append(eval_language_subset)
    else:
        new_datasets_train.append(law_data.select(range(start_index, start_index + 1)))
    start_index = end_index

# Concatenate all new datasets into one
train_data = concatenate_datasets(new_datasets_train)
test_data = concatenate_datasets(new_datasets_test)
eval_data = concatenate_datasets(new_datasets_eval)


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", task=task)

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", task=task)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


model = prepare_model_for_kbit_training(model)
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

eval_dataloader = DataLoader(test_data.select(range(2181,2233)), batch_size=5, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh",task=task)
normalizer = BasicTextNormalizer()

predictions = []
references = []
normalized_predictions = []
normalized_references = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
i=0
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            #print(law_data_backup["sentence"][i])
            #lang_sentence = backup_law_data["sentence"][i]
            i = i + 1
            input_features = batch["input_features"].to(device)
            #forced_decoder_ids = correct_lang_decoder(lang_sentence)
            generated_tokens = (
                    model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].to(device)
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            print(decoded_labels)
            print(decoded_preds)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
            normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
        del generated_tokens, labels, batch, input_features
    gc.collect()
wer = 100 * metric.compute(predictions=predictions, references=references)
normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

print(f"{wer=} and {normalized_wer=}")
print(eval_metrics)

"""
ar:
wer=40.58577405857741 and normalized_wer=43.36569579288026
{'eval/wer': 40.58577405857741, 'eval/normalized_wer': 43.36569579288026}

cs:
wer=11.475409836065573 and normalized_wer=8.19672131147541
{'eval/wer': 11.475409836065573, 'eval/normalized_wer': 8.19672131147541}

de:
wer=4.331528529779258 and normalized_wer=2.617366015787287
{'eval/wer': 4.331528529779258, 'eval/normalized_wer': 2.617366015787287}

el:
wer=20.0 and normalized_wer=6.666666666666667
{'eval/wer': 20.0, 'eval/normalized_wer': 6.666666666666667}

en:
wer=9.177215189873419 and normalized_wer=4.553734061930783
{'eval/wer': 9.177215189873419, 'eval/normalized_wer': 4.553734061930783}

es:
wer=7.203234105108416 and normalized_wer=2.6093348033811097
{'eval/wer': 7.203234105108416, 'eval/normalized_wer': 2.6093348033811097}

fa:
wer=25.04378283712785 and normalized_wer=21.015761821366024
{'eval/wer': 25.04378283712785, 'eval/normalized_wer': 21.015761821366024}

fr:
wer=13.584779706275032 and normalized_wer=9.1533895657607
{'eval/wer': 13.584779706275032, 'eval/normalized_wer': 9.1533895657607}

he:
wer=20.51282051282051 and normalized_wer=7.6923076923076925
{'eval/wer': 20.51282051282051, 'eval/normalized_wer': 7.6923076923076925}

hi:
wer=20.689655172413794 and normalized_wer=2.0408163265306123
{'eval/wer': 20.689655172413794, 'eval/normalized_wer': 2.0408163265306123}

id:
wer=8.75 and normalized_wer=5.0
{'eval/wer': 8.75, 'eval/normalized_wer': 5.0}

it:
wer=10.569948186528498 and normalized_wer=4.805260495700557
{'eval/wer': 10.569948186528498, 'eval/normalized_wer': 4.805260495700557}

ja:
wer=60.0 and normalized_wer=45.0
{'eval/wer': 60.0, 'eval/normalized_wer': 45.0}

ko:-----

nl:
wer=5.682951146560319 and normalized_wer=4.683607374190333
{'eval/wer': 5.682951146560319, 'eval/normalized_wer': 4.683607374190333}

pl:
wer=12.643678160919542 and normalized_wer=1.7241379310344827
{'eval/wer': 12.643678160919542, 'eval/normalized_wer': 1.7241379310344827}

pt:
wer=47.11111111111111 and normalized_wer=17.46724890829694
{'eval/wer': 47.11111111111111, 'eval/normalized_wer': 17.46724890829694}

ro:
wer=9.745762711864407 and normalized_wer=8.403361344537815
{'eval/wer': 9.745762711864407, 'eval/normalized_wer': 8.403361344537815}

ru:
wer=6.163522012578617 and normalized_wer=3.634085213032581
{'eval/wer': 6.163522012578617, 'eval/normalized_wer': 3.634085213032581}

tr:
wer=13.90728476821192 and normalized_wer=9.67741935483871
{'eval/wer': 13.90728476821192, 'eval/normalized_wer': 9.67741935483871}

uk:
wer=18.777292576419214 and normalized_wer=10.572687224669604
{'eval/wer': 18.777292576419214, 'eval/normalized_wer': 10.572687224669604}

vi:
wer=21.428571428571427 and normalized_wer=0.0
{'eval/wer': 21.428571428571427, 'eval/normalized_wer': 0.0}

zh-CN:
wer=85.18518518518519 and normalized_wer=62.857142857142854
{'eval/wer': 85.18518518518519, 'eval/normalized_wer': 62.857142857142854}
"""
