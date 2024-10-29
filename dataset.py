import pandas as pd
import torch
from utils import match_tokens_labels

class NerDataset(torch.utils.data.Dataset):
  """
  Custom dataset implementation to get (text,labels) tuples
  Inputs:
   - df : dataframe with columns [tags, sentence]
  """

  def __init__(self, df, idx2tag, tokenizer):
    if not isinstance(df, pd.DataFrame):
      raise TypeError('Input should be a dataframe')

    if "tags" not in df.columns or "sentence" not in df.columns:
      raise ValueError("Dataframe should contain 'tags' and 'sentence' columns")

    tags_list = [i.split() for i in df["tags"].values.tolist()]
    texts = df["sentence"].values.tolist()

    self.texts = [tokenizer(text, padding = "max_length", max_length = 512, truncation = True, return_tensors = "pt") for text in texts]
    self.labels = [match_tokens_labels(text, tags, idx2tag) for text,tags in zip(self.texts, tags_list)]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_text = self.texts[idx]
    batch_labels = self.labels[idx]

    return batch_text, torch.LongTensor(batch_labels)