import pandas as pd

def tags_2_labels(tags : str, tag2idx : dict, unseen_label: str):
  '''
  Method that takes a list of tags and a dictionary mapping and returns a list of labels (associated).
  Used to create the "label" column in df from the "tags" column.
  '''
  return [tag2idx[tag] if tag in tag2idx else unseen_label for tag in tags.split()]


def tags_mapping(df_train : pd.DataFrame, tags_series : pd.Series):
  """
  tag_series = df column with tags for each sentence.
  Returns:
    - dictionary mapping tags to indexes (label)
    - dictionary mappign inedexes to tags
    - The label corresponding to tag 'O'
    - A set of unique tags ecountered in the trainind df, this will define the classifier dimension
  """

  if not isinstance(tags_series, pd.Series):
      raise TypeError('Input should be a padas Series')

  unique_tags = set()

  for tag_list in df_train["tags"]:
    for tag in tag_list.split():
      unique_tags.add(tag)


  tag2idx = {k:v for v,k in enumerate(sorted(unique_tags))}
  idx2tag = {k:v for v,k in tag2idx.items()}

  unseen_label = tag2idx["O"]

  return tag2idx, idx2tag, unseen_label, unique_tags

def match_tokens_labels(tokenized_input, tags, tag2idx, ignore_token = -100):
        '''
        Used in the custom dataset.
        -100 will be tha label used to match additional tokens like [CLS] [PAD] that we dont care about.
        Inputs :
          - tokenized_input : tokenizer over the imput text -> {input_ids, attention_mask}
          - tags : is a single label array -> [O O O O O O O O O O O O O O B-tim O]

        Returns a list of labels that match the tokenized text -> [-100, 3,5,6,-100,...]
        '''

        #gives an array [ None , 0 , 1 ,2 ,... None]. Each index tells the word of reference of the token
        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(ignore_token)

            #if its equal to the previous word we can add the same label id of the provious or -100
            else :
                try:
                  reference_tag = tags[word_idx]
                  label_ids.append(tag2idx[reference_tag])
                except:
                  label_ids.append(ignore_token)

            previous_word_idx = word_idx

        return label_ids

def get_labels_unique_word(tokenized_input, predictions, idx2tag):
    word_ids = tokenized_input.word_ids()
    
    previous_word_idx = -1
    unique_tags_pred = []

    for word_idx in word_ids:
        if word_idx is None or (word_idx == previous_word_idx):
            continue

        else:
            idx_in_word_ids_array = word_ids.index(word_idx)
            reference_tag = predictions[idx_in_word_ids_array]
            unique_tags_pred.append(idx2tag[reference_tag])

        previous_word_idx = word_idx
  
    return unique_tags_pred

def retrieve_token_tag_and_tag_pred(text_tokenized, predictions, dev_label, idx2tag):
    word_ids = text_tokenized.word_ids()
    previous_index = None

    retrieved_tags_pred = []
    retrieved_tags_dev = []
    i = 0
    predictions = predictions[0]
    dev_label = dev_label[0]

    for word_idx in word_ids:
        if word_idx == None:
            pass
        elif word_idx == previous_index:
            pass
        else:
            retrieved_tags_pred.append(idx2tag[predictions[i]])
            if dev_label[i] == -100 or dev_label[i] == "-100":
                retrieved_tags_dev.append("O")
            else:
                retrieved_tags_dev.append(idx2tag[dev_label[i]])

        i += 1
        previous_index = word_idx

    return retrieved_tags_dev, retrieved_tags_pred