import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from model.multimodal_bert import MultimodalBERT
from preprocess.preprocess import Preprocessor
from core.settings import TransformerSettings


class PhishingDataset(Dataset):
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url = self.data.iloc[idx]["url"]
        html_content = self.data.iloc[idx]["html_content"]
        label = 1 if self.data.iloc[idx]["label"] == "malicious" else 0

        # NaN -> ""
        html_content = "" if pd.isna(html_content) else html_content

        # preprocess URL, HTML
        url_input_ids, url_attention_mask = self.preprocessor.preprocess_url(url)
        html_input_ids, html_attention_mask = self.preprocessor.preprocess_html(html_content)

        return {
            "url_input_ids": url_input_ids.squeeze(),
            "url_attention_mask": url_attention_mask.squeeze(),
            "html_input_ids": html_input_ids.squeeze(),
            "html_attention_mask": html_attention_mask.squeeze(),
            "label": torch.tensor(label, dtype=torch.float)
        }



def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            batch["url_input_ids"], batch["url_attention_mask"],
            batch["html_input_ids"], batch["html_attention_mask"]
        ).squeeze()

        loss = criterion(outputs, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["url_input_ids"], batch["url_attention_mask"],
                batch["html_input_ids"], batch["html_attention_mask"]
            ).squeeze()

            preds = torch.round(torch.sigmoid(outputs))
            predictions.extend(preds.tolist())
            labels.extend(batch["label"].tolist())

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return acc, f1


def main():
    settings = TransformerSettings()
    preprocessor = Preprocessor(max_token_len=settings.MAX_TOKEN_LEN)

    data = pd.read_csv("src/phishing_data.csv", header=None, names=["id", "url", "html_content", "label"])
    train_data = data.sample(frac=0.8, random_state=42)
    val_data = data.drop(train_data.index)

    train_dataset = PhishingDataset(train_data, preprocessor)
    val_dataset = PhishingDataset(val_data, preprocessor)

    train_dataloader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE)

    # 최적화
    model = MultimodalBERT(embedding_dim=settings.EMBEDDING_DIM)
    optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 학습 루프
    for epoch in range(settings.EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, criterion)
        val_acc, val_f1 = evaluate(model, val_dataloader)
        print(
            f"Epoch {epoch + 1}/{settings.EPOCHS}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")


if __name__ == "__main__":
    main()
