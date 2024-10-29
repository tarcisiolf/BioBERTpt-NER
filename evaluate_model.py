import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from dataset import NerDataset
from metrics import MetricsTracking
from utils import retrieve_token_tag_and_tag_pred

def evaluate_test_texts(model, df_test, tokenizer, batch_size = 1):

    dev_dataset = NerDataset(df_test)
    dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dev_metrics = MetricsTracking()
    total_loss_dev = 0

    i = 0

    text_labels_dev = []
    text_labels_pred = []
    with torch.no_grad():
        for dev_data, dev_label in dev_dataloader:
            dev_label = dev_label.to(device)

            mask = dev_data['attention_mask'].squeeze(1).to(device)
            input_id = dev_data['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask, dev_label)
            loss, logits = output.loss, output.logits
        
            predictions = logits.argmax(dim= -1)

            tag = df_test.tags.iloc[i]
            text = df_test.sentence.iloc[i]

            text_tokenized = tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
            labels_dev, labels_pred = retrieve_token_tag_and_tag_pred(text, text_tokenized, tag, predictions.tolist(), dev_label.tolist(), tokenizer)
            text_labels_dev.append(labels_dev)
            text_labels_pred.append(labels_pred)


            dev_metrics.update(predictions, dev_label)
            total_loss_dev += loss.item()
            i += 1
            

    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))

    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" )


    return text_labels_dev, text_labels_pred

def create_classification_report(dev, pred):
    result_dict = classification_report(dev, pred, mode="strict", scheme=IOB2, zero_division=False)
    print(result_dict)