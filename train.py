from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from metrics import MetricsTracking

def train_loop_wt_eval(model, train_dataset, optimizer,  batch_size, epochs):

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs) :

        train_metrics = MetricsTracking()
        total_loss_train = 0

        model.train() #train mode

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            '''
            squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len]
            '''
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()

            output = model(input_id, mask, train_label)
            loss, logits = output.loss, output.logits
            predictions = logits.argmax(dim= -1)

            #compute metrics
            train_metrics.update(predictions, train_label)
            total_loss_train += loss.item()

            #grad step
            loss.backward()
            optimizer.step()

    train_results = train_metrics.return_avg_metrics(len(train_dataloader))

    print(f"TRAIN \nLoss: {total_loss_train / len(train_dataset)} \nMetrics {train_results}\n" )