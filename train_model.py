import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.optim import SGD

from dataset import NerDataset
from utils import tags_mapping
from utils import tags_2_labels
from model import create_model
from train import train_loop_wt_eval


file_path = 'data/df_tokens_labeled_iob_bert_format.csv'
df_total = pd.read_csv(file_path, encoding='utf-8')

df_total.rename(columns = {'text':'sentence', 'iob_labels':'tags'}, inplace = True)

train_file_path = 'data/df_train_llms_tokens_labeled_iob_bert_format.csv'
test_file_path = 'data/df_test_llms_tokens_labeled_iob_bert_format.csv'

df_train = pd.read_csv(train_file_path, encoding='utf-8')
df_test = pd.read_csv(test_file_path, encoding='utf-8')

df_train.rename(columns = {'text':'sentence', 'iob_labels':'tags'}, inplace = True)
df_test.rename(columns = {'text':'sentence', 'iob_labels':'tags'}, inplace = True)

#create tag-label mapping
#tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(df_train["tags"])
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping(df_total, df_total["tags"])

#create the label column from tag. Unseen labels will be tagged as "O"
for df in [df_train, df_test]:
  df["labels"] = df["tags"].apply(lambda tags : tags_2_labels(tags, tag2idx, unseen_label))

#original text
text = df_total["sentence"].values.tolist()

#toeknized text
tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-all", do_lower_case=False)
text_tokenized = tokenizer(text , padding = "max_length", max_length = 512, truncation = True, return_tensors = "pt" )

#mapping token to original word
word_ids = text_tokenized.word_ids()

#datasets
train_dataset = NerDataset(df_train, idx2tag, tokenizer)

#create model   
model = create_model(unique_tags)
learning_rate = 1e-3
optimizer = SGD(model.parameters(), lr=learning_rate, momentum = 0.9)

parameters = {
    "model": model,
    "train_dataset": train_dataset,
    "optimizer" : optimizer,
    "batch_size" : 4,
    "epochs" : 1
}


#train the model
train_loop_wt_eval(**parameters)

"""
#save the model
model_save_path = 'models\model_biobert_llms_iob_format.pth'
model.save(model.state_dict(), model_save_path)

print(f"Model saved at {model_save_path}")
"""