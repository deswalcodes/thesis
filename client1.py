


import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from opacus import PrivacyEngine
from sklearn.feature_extraction.text import TfidfVectorizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_dataset(file_path, client_id, total_clients=2):
    with open('Sentences_AllAgree.txt', 'r', encoding='latin1') as file:
        lines = file.readlines()

    texts = []
    labels = []
    for line in lines:
        text, label = line.strip().split('@')
        texts.append(text)
        labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(texts).toarray()

    # Split data between clients
    data_size = len(features)
    client_data_size = data_size // total_clients
    start_idx = (client_id - 1) * client_data_size
    end_idx = client_id * client_data_size if client_id != total_clients else data_size

    X_client = features[start_idx:end_idx]
    y_client = labels[start_idx:end_idx]

    X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
print("ok")

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Increased units
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)         # New hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


print("okk")

def get_dataloader(X, y, batch_size=8):
    dataset = TextDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


print("okkk")

# Load Client 1 Data
client_id = 1
X_train, y_train, X_test, y_test = load_dataset('Sentences_AllAgree.txt', client_id)

# Prepare DataLoader
train_loader = get_dataloader(X_train, y_train)

# Model, Optimizer, Loss
input_size = X_train.shape[1]
num_classes = len(set(y_train))
model = SimpleClassifier(input_size, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()

# Apply Differential Privacy
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.2,
    max_grad_norm=1.0,
)

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, train_loader, criterion, optimizer, device, epochs=5)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        test_loader = get_dataloader(X_test, y_test, batch_size=32)

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        return float(loss), len(test_loader.dataset), {"accuracy": float(accuracy)}

# Start the client
fl.client.start_numpy_client(server_address="localhost:8081", client=FlowerClient())
print("all")