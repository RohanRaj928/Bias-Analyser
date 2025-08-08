import multiprocessing

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def main():

    import pathlib

    import pandas as pd
    from torch.utils.data import DataLoader
    from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
    import torch
    import tqdm  # Progress Bar

    # Training Variables Default
    num_epochs = 3
    batch_size = 2
    learning_rate = 1e-5
    num_workers = 1

    base_dir = pathlib.Path(__file__).resolve().parent.parent
    filepath = base_dir / 'Models' / 'Classification'

    # Set correct device, Override if wanted
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    device = torch.device('cpu')

    print("Using device:", device.type)


    print("Importing Dataset...")
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'valid': 'data/valid-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/siddharthmb/article-bias-prediction-media-splits/" + splits["train"])


    print("Preparing for training...")
    # 0: Left, 1: Right , 2: Center
    text = df['content'].tolist()
    labels = df['bias_text'].tolist()


    #Load Tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    encodings = tokenizer(text, padding=True, truncation=True, max_length=4096, return_tensors='pt') # Tokenize text
    dataset = Dataset(encodings, labels)

    #Load Model
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=3)
    model.to(device)

    # Save the model
    print(f"Saving model to {filepath}...")
    filepath.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(filepath)
    tokenizer.save_pretrained(filepath)
    print("Model saved.")

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    print("Training...")
    model.train()

    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0

        for batch in tqdm.tqdm(train_loader, leave=True):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}


            # Use AMP if available
            if device.type == 'cuda':
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Fallback
            else:

                optimizer.zero_grad()

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}\n")


    # Save the model
    print(f"Saving model to {filepath}...")
    filepath.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(filepath)
    tokenizer.save_pretrained(filepath)
    print("Model saved.")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Needed if being converted to executable
    main()

