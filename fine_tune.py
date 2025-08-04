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

login(token = "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",add_to_git_credential=True)

model_name_or_path = "openai/whisper-large-v3"
#model_name_or_path = "fine-tuned-ar"
#model_name_or_path = "fine-tuned"

#model_name_or_path="devasheeshG/whisper_large_v2_fp16_transformers"

#model_name_or_path="airesearch/wav2vec2-large-xlsr-53-th"
task = "transcribe"

law_data = datasets.load_from_disk("zh_health_dataset")

from datasets import Dataset, concatenate_datasets

# Example combined dataset (replace this with your actual dataset)
# combined_dataset = ...  # Load your actual dataset here

# Sentence counts for each language
#language numbers for health
sentence_counts = [1288, 416, 1843, 33, 3739, 3602, 1583, 2227, 58, 116, 177, 2574, 187, 13, 919, 276, 637, 130, 679, 666,556, 69, 516]
#language numbers for law
#sentence_counts = [470, 365, 2543, 16, 3869, 2724, 881, 2979, 38, 43, 89, 1974, 103, 1, 1942, 355, 551, 334, 870, 251,361, 8, 274]

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

print(len(test_data))

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)

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


model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit= True, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


model = prepare_model_for_kbit_training(model)
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="reach-vb/test",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    eval_steps=100,  # Evaluate every 100 steps
    gradient_checkpointing=True,
    num_train_epochs=1,
    evaluation_strategy="steps",
    fp16=True,
    report_to="none",
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=1,
    lr_scheduler_type="linear",
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    greater_is_better=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
   # callbacks=[SavePeftModelCallback],
)
torch.cuda.empty_cache()
#model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.model.save_pretrained("fine-tuned-zh")
trainer.tokenizer.save_pretrained("fine-tuned-zh")
#'train_loss': 2.2751105626424155

#***************************************************************************
#evaluation
peft_model_id = "fine-tuned-zh" # Use the same model ID as before.
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_model_id)
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,  device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
model.config.use_cache = True


eval_dataloader = DataLoader(test_data, batch_size=5, collate_fn=data_collator)
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
ar only
wer=41.84100418410041 and normalized_wer=43.36569579288026
{'eval/wer': 41.84100418410041, 'eval/normalized_wer': 43.36569579288026}

cs only
wer=10.491803278688524 and normalized_wer=6.885245901639345
{'eval/wer': 10.491803278688524, 'eval/normalized_wer': 6.885245901639345}

de only
wer=3.998334027488547 and normalized_wer=2.5758205234732032
{'eval/wer': 3.998334027488547, 'eval/normalized_wer': 2.5758205234732032}

el only
wer=20.0 and normalized_wer=6.666666666666667
{'eval/wer': 20.0, 'eval/normalized_wer': 6.666666666666667}

en only
wer=8.887130801687764 and normalized_wer=4.579755399427531
{'eval/wer': 8.887130801687764, 'eval/normalized_wer': 4.579755399427531}

es only
wer=7.092980521866961 and normalized_wer=2.4623300257258363
{'eval/wer': 7.092980521866961, 'eval/normalized_wer': 2.4623300257258363}

fa only----????????????
wer=25.04378283712785 and normalized_wer=21.015761821366024
{'eval/wer': 25.04378283712785, 'eval/normalized_wer': 21.015761821366024}

fr only
wer=13.484646194926569 and normalized_wer=8.965948141205873
{'eval/wer': 13.484646194926569, 'eval/normalized_wer': 8.965948141205873}

he only:
wer=20.51282051282051 and normalized_wer=7.6923076923076925
{'eval/wer': 20.51282051282051, 'eval/normalized_wer': 7.6923076923076925}

hi only:
wer=20.689655172413794 and normalized_wer=2.0408163265306123
{'eval/wer': 20.689655172413794, 'eval/normalized_wer': 2.0408163265306123}

id only:
wer=8.75 and normalized_wer=5.0
{'eval/wer': 8.75, 'eval/normalized_wer': 5.0}

it only:
wer=10.414507772020725 and normalized_wer=4.704097116843703
{'eval/wer': 10.414507772020725, 'eval/normalized_wer': 4.704097116843703}

ja only:
wer=60.0 and normalized_wer=45.0
{'eval/wer': 60.0, 'eval/normalized_wer': 45.0}

nl only:
wer=5.8325024925224325 and normalized_wer=4.733432984554061
{'eval/wer': 5.8325024925224325, 'eval/normalized_wer': 4.733432984554061}

pl only:
wer=12.35632183908046 and normalized_wer=1.4367816091954022
{'eval/wer': 12.35632183908046, 'eval/normalized_wer': 1.4367816091954022}

pt only:
wer=45.33333333333333 and normalized_wer=15.283842794759824
{'eval/wer': 45.33333333333333, 'eval/normalized_wer': 15.283842794759824}


ro only:
wer=9.745762711864407 and normalized_wer=8.403361344537815
{'eval/wer': 9.745762711864407, 'eval/normalized_wer': 8.403361344537815}

ru only:
wer=6.289308176100629 and normalized_wer=3.634085213032581
{'eval/wer': 6.289308176100629, 'eval/normalized_wer': 3.634085213032581}

tr only:
wer=13.90728476821192 and normalized_wer=9.67741935483871
{'eval/wer': 13.90728476821192, 'eval/normalized_wer': 9.67741935483871}

uk only:
wer=18.340611353711793 and normalized_wer=9.691629955947137
{'eval/wer': 18.340611353711793, 'eval/normalized_wer': 9.691629955947137}

vi only:
wer=14.285714285714285 and normalized_wer=0.0
{'eval/wer': 14.285714285714285, 'eval/normalized_wer': 0.0}

zh-CN only:
wer=88.88888888888889 and normalized_wer=62.857142857142854
{'eval/wer': 88.88888888888889, 'eval/normalized_wer': 62.857142857142854}

************************fine-tuned******************************************
ar: 
wer=43.93305439330544 and normalized_wer=43.36569579288026
{'eval/wer': 43.93305439330544, 'eval/normalized_wer': 43.36569579288026}

cs:
wer=10.819672131147541 and normalized_wer=7.540983606557377
{'eval/wer': 10.819672131147541, 'eval/normalized_wer': 7.540983606557377}

de:
wer=4.539775093710953 and normalized_wer=2.9912754466140425
{'eval/wer': 4.539775093710953, 'eval/normalized_wer': 2.9912754466140425}

el:
wer=20.0 and normalized_wer=6.666666666666667
{'eval/wer': 20.0, 'eval/normalized_wer': 6.666666666666667}

en:
wer=8.966244725738397 and normalized_wer=4.657819411917773
{'eval/wer': 8.966244725738397, 'eval/normalized_wer': 4.657819411917773}

es:
wer=7.239985299522235 and normalized_wer=2.499081220139655
{'eval/wer': 7.239985299522235, 'eval/normalized_wer': 2.499081220139655}

fa:
wer=21.541155866900176 and normalized_wer=19.614711033274958
{'eval/wer': 21.541155866900176, 'eval/normalized_wer': 19.614711033274958}

fr:
wer=13.718291054739653 and normalized_wer=9.309590752889722
{'eval/wer': 13.718291054739653, 'eval/normalized_wer': 9.309590752889722}

he:
wer=28.205128205128204 and normalized_wer=17.94871794871795
{'eval/wer': 28.205128205128204, 'eval/normalized_wer': 17.94871794871795}

hi:
wer=27.586206896551722 and normalized_wer=4.081632653061225
{'eval/wer': 27.586206896551722, 'eval/normalized_wer': 4.081632653061225}

id:
wer=6.25 and normalized_wer=2.5
{'eval/wer': 6.25, 'eval/normalized_wer': 2.5}

it:
wer=11.55440414507772 and normalized_wer=4.299443601416288
{'eval/wer': 11.55440414507772, 'eval/normalized_wer': 4.299443601416288}

ja:
wer=80.0 and normalized_wer=75.0
{'eval/wer': 80.0, 'eval/normalized_wer': 75.0}

ko: we cannot test korean there isn't sufficient data we used in train

nl:
wer=6.281156530408774 and normalized_wer=4.633781763826607
{'eval/wer': 6.281156530408774, 'eval/normalized_wer': 4.633781763826607}

pl:
wer=12.931034482758621 and normalized_wer=2.2988505747126435
{'eval/wer': 12.931034482758621, 'eval/normalized_wer': 2.2988505747126435}

pt:
wer=39.55555555555556 and normalized_wer=14.847161572052403
{'eval/wer': 39.55555555555556, 'eval/normalized_wer': 14.847161572052403}

ro:
wer=11.440677966101696 and normalized_wer=9.243697478991598
{'eval/wer': 11.440677966101696, 'eval/normalized_wer': 9.243697478991598}

ru:
wer=5.911949685534592 and normalized_wer=3.258145363408521
{'eval/wer': 5.911949685534592, 'eval/normalized_wer': 3.258145363408521}

tr:
wer=19.205298013245034 and normalized_wer=15.483870967741936
{'eval/wer': 19.205298013245034, 'eval/normalized_wer': 15.483870967741936}

uk:
wer=16.593886462882097 and normalized_wer=7.048458149779736
{'eval/wer': 16.593886462882097, 'eval/normalized_wer': 7.048458149779736}

vi:
wer=14.285714285714285 and normalized_wer=0.0
{'eval/wer': 14.285714285714285, 'eval/normalized_wer': 0.0}

zh-CN:
wer=85.18518518518519 and normalized_wer=71.42857142857143
{'eval/wer': 85.18518518518519, 'eval/normalized_wer': 71.42857142857143}

*************************** health dataset ******************************************

ar-only:
wer=35.69277108433735 and normalized_wer=29.7554347826087
{'eval/wer': 35.69277108433735, 'eval/normalized_wer': 29.7554347826087}

cs-only:
wer=13.084112149532709 and normalized_wer=11.526479750778815
{'eval/wer': 13.084112149532709, 'eval/normalized_wer': 11.526479750778815}

de-only:
wer=4.545454545454546 and normalized_wer=3.5756154747948417
{'eval/wer': 4.545454545454546, 'eval/normalized_wer': 3.5756154747948417}

el-only:
wer=22.22222222222222 and normalized_wer=7.4074074074074066
{'eval/wer': 22.22222222222222, 'eval/normalized_wer': 7.4074074074074066}

en-only:
wer=7.616066770996348 and normalized_wer=4.344473007712082
{'eval/wer': 7.616066770996348, 'eval/normalized_wer': 4.344473007712082}

es-only:
wer=6.3776223776223775 and normalized_wer=2.8019052956010086
{'eval/wer': 6.3776223776223775, 'eval/normalized_wer': 2.8019052956010086}

fa-only:
{'eval/wer': 27.505567928730514, 'eval/normalized_wer': 25.16703786191537}

fr-only:
wer=8.27165868524162 and normalized_wer=5.660377358490567
{'eval/wer': 8.27165868524162, 'eval/normalized_wer': 5.660377358490567}

he-only:
wer=29.09090909090909 and normalized_wer=16.363636363636363
{'eval/wer': 29.09090909090909, 'eval/normalized_wer': 16.363636363636363}

hi-only:
wer=21.649484536082475 and normalized_wer=4.651162790697675
{'eval/wer': 21.649484536082475, 'eval/normalized_wer': 4.651162790697675}

id-only:
wer=7.602339181286549 and normalized_wer=5.780346820809249
{'eval/wer': 7.602339181286549, 'eval/normalized_wer': 5.780346820809249}

it-only:
wer=7.786728039892597 and normalized_wer=4.066265060240964
{'eval/wer': 7.786728039892597, 'eval/normalized_wer': 4.066265060240964}

ja-only:
wer=84.21052631578947 and normalized_wer=83.33333333333334
{'eval/wer': 84.21052631578947, 'eval/normalized_wer': 83.33333333333334}

ko-only:
wer=0.0 and normalized_wer=0.0
{'eval/wer': 0.0, 'eval/normalized_wer': 0.0}

nl-only:
wer=2.793296089385475 and normalized_wer=2.0089285714285716
{'eval/wer': 2.793296089385475, 'eval/normalized_wer': 2.0089285714285716}

pl-only:
wer=13.011152416356877 and normalized_wer=1.858736059479554
{'eval/wer': 13.011152416356877, 'eval/normalized_wer': 1.858736059479554}

pt-only:
wer=17.042606516290725 and normalized_wer=2.5
{'eval/wer': 17.042606516290725, 'eval/normalized_wer': 2.5}

ro-only:
wer=11.224489795918368 and normalized_wer=7.216494845360824
{'eval/wer': 11.224489795918368, 'eval/normalized_wer': 7.216494845360824}

ru-only:
wer=3.054662379421222 and normalized_wer=2.07667731629393
{'eval/wer': 3.054662379421222, 'eval/normalized_wer': 2.07667731629393}

tr-only:
wer=10.192837465564738 and normalized_wer=7.103825136612022
{'eval/wer': 10.192837465564738, 'eval/normalized_wer': 7.103825136612022}

uk-only:
wer=16.666666666666664 and normalized_wer=11.634349030470915
{'eval/wer': 16.666666666666664, 'eval/normalized_wer': 11.634349030470915}

vi-only:
wer=28.846153846153843 and normalized_wer=7.6923076923076925
{'eval/wer': 28.846153846153843, 'eval/normalized_wer': 7.6923076923076925}

zh-only:
wer=88.46153846153845 and normalized_wer=59.64912280701754
{'eval/wer': 88.46153846153845, 'eval/normalized_wer': 59.64912280701754}
"""
