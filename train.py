import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, f1_score
import preprocess
from dataset import SentimentDataset

def train(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5):
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Validation F1-Score: {val_f1:.4f}")

def main():
    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Using CPU only")


    data_file = "data/training.1600000.processed.noemoticon.csv"
    print("Loading and preprocessing data...")
    df = preprocess.load_and_preprocess_data(data_file)
    
    print("Splitting data...")
    train_texts, val_texts, train_labels, val_labels = preprocess.split_data(df)
    
    print("Tokenizing data...")
    train_encodings, val_encodings = preprocess.tokenize_data(train_texts, val_texts)
    
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    print("Starting training...")
    train(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5)

    model.save_pretrained("sentiment140-bert-model")
    print("Model saved to 'sentiment140-bert-model'")

if __name__ == "__main__":
    main()