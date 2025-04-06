import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, TensorDataset
from helper import get_BERT_train_test_data

X_train, X_test, y_train, y_test = get_BERT_train_test_data()

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Load the pretrained BERT tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize the sentences
def tokenize_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

train_encodings = tokenize_sentences(X_train)
test_encodings = tokenize_sentences(X_test)

# Convert to PyTorch Dataset
train_inputs = train_encodings['input_ids']
train_masks = train_encodings['attention_mask']
train_labels = torch.tensor(y_train)

test_inputs = test_encodings['input_ids']
test_masks = test_encodings['attention_mask']
test_labels = torch.tensor(y_test)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
test_data = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the BERT-based model with the classifier on top
class BertForEmotionAnalysis(nn.Module):
    def __init__(self):
        super(BertForEmotionAnalysis, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(768, 256)  # First fully connected layer (BERT output size: 768)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(256, 4)  # Output layer (4 classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]  # Get the representation of the [CLS] token
        
        x = self.fc1(pooled_output)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
model = BertForEmotionAnalysis()

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids, attention_mask)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
