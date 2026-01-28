import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# 1️⃣ Dataset class (same as training)
# -----------------------------
class CleanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                if 'malignant' in item.lower():
                    binary_label = 1
                elif 'benign' in item.lower():
                    binary_label = 0
                else:
                    continue
                
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(item_path, filename)
                        try:
                            with Image.open(image_path) as img:
                                if img.size[0] >= 50 and img.size[1] >= 50:
                                    self.images.append(image_path)
                                    self.labels.append(binary_label)
                        except:
                            continue
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# -----------------------------
# 2️⃣ Transforms (same as validation)
# -----------------------------
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------------
# 3️⃣ Model class (must match training)
# -----------------------------
import timm
import torch.nn as nn

class CleanEnhancedMobileNetV3(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0, global_pool='avg')
        with torch.no_grad():
            dummy_input = torch.randn(1,3,224,224)
            feature_dim = self.backbone(dummy_input).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# -----------------------------
# 4️⃣ Load model checkpoint
# -----------------------------
device = torch.device('cpu')
model = CleanEnhancedMobileNetV3(dropout=0.2).to(device)

checkpoint_path = "/content/drive/MyDrive/skin_cancer_phone_project/mobilenetv3_minimal_lr_bump.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded successfully")

# -----------------------------
# 5️⃣ Prepare validation loader
# -----------------------------
val_dir = '/content/drive/MyDrive/skin_cancer_phone_project/datasets/synthetic_dataset/validation'
val_dataset = CleanDataset(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 6️⃣ Compute metrics
# -----------------------------
criterion = nn.BCEWithLogitsLoss()
all_preds = []
all_targets = []
total_loss = 0.0

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data).squeeze()
        loss = criterion(output, target)
        total_loss += loss.item()
        preds = (torch.sigmoid(output) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

val_loss = total_loss / len(val_loader)
val_acc = sum([p==t for p,t in zip(all_preds, all_targets)]) / len(all_targets)
val_precision = precision_score(all_targets, all_preds, zero_division=0)
val_recall = recall_score(all_targets, all_preds, zero_division=0)
val_f1 = f1_score(all_targets, all_preds, zero_division=0)

print("\n📊 Validation Results:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_acc:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1 Score: {val_f1:.4f}")