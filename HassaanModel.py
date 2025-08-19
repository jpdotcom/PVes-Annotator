import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import os
from itertools import cycle
from sklearn.utils.class_weight import compute_class_weight


# Set working directory
os.chdir(r'H:\Postdoc data Margot Lab\data_from_Steve\pves-classifier-main\PVES-classifier-main\PVES-classifier-main\Classification')

# Load dataset
dataset = pd.read_csv('training_data_multi_label_balanced_with_class.csv')
features = [
    "cA1_max", "cA1_mav", "cA1_med", "cA1_ent",
    "cD1_max", "cD1_mav", "cD1_med", "cD1_ent",
    "xcorr1_med", "xcorr1_mean", "xcorr1_max",
    "cA2_max", "cA2_mav", "cA2_med", "cA2_ent",
    "cD2_max", "cD2_mav", "cD2_med", "cD2_ent",
    "xcorr2_med", "xcorr2_mean", "xcorr2_max",
    "cA3_max", "cA3_mav", "cA3_med", "cA3_ent",
    "cD3_max", "cD3_mav", "cD3_med", "cD3_ent",
    "xcorr3_med", "xcorr3_mean", "xcorr3_max",
    "cA4_max", "cA4_mav", "cA4_med", "cA4_ent",
    "cD4_max", "cD4_mav", "cD4_med", "cD4_ent",
    "xcorr4_med", "xcorr4_mean", "xcorr4_max",
    "cA5_max", "cA5_mav", "cA5_med", "cA5_ent",
    "cD5_max", "cD5_mav", "cD5_med", "cD5_ent",
    "xcorr5_med", "xcorr5_mean", "xcorr5_max"
]

X = dataset[features].to_numpy()
y = dataset['class']
y_onehot = label_binarize(y, classes=['abd', 'do', 'none', 'void'])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stage 1 labels: 1 = NONE, 0 = OTHERS
y_stage1 = np.array([1 if np.argmax(row) == 2 else 0 for row in y_onehot])
y_class_index = np.array([np.argmax(row) for row in y_onehot])
# Stage 1 class labels
class_labels_stage1 = ['EVENT', 'NONE']  # row 0 = EVENT, row 1 = NONE

# Dataset class
class UDSDataset(Dataset):
    def __init__(self, X_all, y_all):
        self.X = torch.tensor(X_all, dtype=torch.float32)
        self.y = torch.tensor(y_all, dtype=torch.long)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.X.shape[0]

# Neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Hyperparameters
input_size = X.shape[1]
hidden1, hidden2 = 200, 200
batch_size = 200
epochs = 100
lr = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
patience = 15 #5,10,15,20,25,30, especially for early stopping

# Storage
y_pred_stage1, y_actual_stage1 = [], []
y_probs_stage1_master = []  # global Stage-1 probability storage
y_pred_stage2, y_actual_stage2 = [], []
y_probs_stage2_master = []  # global Stage-2 probability storage
dataset['class_predicted'] = None
fold = 0

# Learning curve storage
train_losses_s1_folds, val_losses_s1_folds = [], []
train_losses_s2_folds, val_losses_s2_folds = [], []

# ROC storage
roc_auc_stage1 = np.zeros((2,5))
roc_auc_stage2 = np.zeros((3,5))
fpr_stage1 = [[] for _ in range(2)]
tpr_stage1 = [[] for _ in range(2)]
fpr_stage2 = [[] for _ in range(3)]
tpr_stage2 = [[] for _ in range(3)]

# Main cross-validation loop
for train_idx, test_idx in kf.split(X):
    # === Stage 1 ===
    model_s1 = NeuralNet(input_size, hidden1, hidden2, 2).to(device)
    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=lr)
#    crit = nn.CrossEntropyLoss()
    class_weights_s1 = compute_class_weight('balanced', classes=np.unique(y_stage1[train_idx]), y=y_stage1[train_idx])
    crit_s1 = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_s1, dtype=torch.float32).to(device))

    # Scale inside fold to avoid leakage
    scaler_fold = StandardScaler().fit(X[train_idx])
    X_train_fold = scaler_fold.transform(X[train_idx])
    X_test_fold = scaler_fold.transform(X[test_idx])

    train_loader_s1 = DataLoader(UDSDataset(X_train_fold, y_stage1[train_idx]), batch_size=batch_size, shuffle=True)
    test_loader_s1 = DataLoader(UDSDataset(X_test_fold, y_stage1[test_idx]), batch_size=batch_size, shuffle=False)

    train_losses_s1, val_losses_s1 = [], []
    best_val_loss_s1 = float('inf')
    best_model_s1_state = None
    counter_s1 = 0

    for epoch in range(epochs):
        # Training
        model_s1.train()
        epoch_loss = 0
        for xb, yb in train_loader_s1:
            xb, yb = xb.to(device), yb.to(device)
            out = model_s1(xb)
            loss = crit_s1(out, yb)
            opt_s1.zero_grad()
            loss.backward()
            opt_s1.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader_s1.dataset)
        train_losses_s1.append(epoch_loss)

        # Validation
        model_s1.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader_s1:
                xb, yb = xb.to(device), yb.to(device)
                out = model_s1(xb)
                loss = crit_s1(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(test_loader_s1.dataset)
        val_losses_s1.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss_s1:
            best_val_loss_s1 = val_loss
            best_model_s1_state = model_s1.state_dict()
            counter_s1 = 0
        else:
            counter_s1 += 1
            if counter_s1 >= patience:
                print(f"Stage 1 Early stopping at epoch {epoch+1}, fold {fold+1}")
                model_s1.load_state_dict(best_model_s1_state)
                break

    train_losses_s1_folds.append(train_losses_s1)
    val_losses_s1_folds.append(val_losses_s1)

    # Stage 1 evaluation
    model_s1.eval()
    y_probs_stage1_fold = []
    y_pred_stage1_fold, y_actual_stage1_fold = [], []

    with torch.no_grad():
        for xb, yb in test_loader_s1:
            xb = xb.to(device)
            out = model_s1(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_pred_stage1_fold.extend(preds.tolist())
            y_actual_stage1_fold.extend(yb.numpy().tolist())
            y_probs_stage1_fold.extend(probs[:,1])  # prob of class "1" (NONE)

    # Append once per fold
    y_pred_stage1.extend(y_pred_stage1_fold)
    y_actual_stage1.extend(y_actual_stage1_fold)
    y_probs_stage1_master.extend(y_probs_stage1_fold)

    print(f"Fold {fold+1} S1: y_true={len(y_actual_stage1_fold)}, y_pred={len(y_pred_stage1_fold)}")

    # Stage 1 ROC
    probs_all = np.vstack([
        F.softmax(model_s1(xb.to(device)), dim=1).detach().cpu().numpy()
        for xb, yb in test_loader_s1
    ])
    
    for cls in [0, 1]:  # EVENT=0, NONE=1
        # Only compute ROC if this class exists in y_actual_stage1_fold
        if cls in y_actual_stage1_fold:
            y_true_bin = (np.array(y_actual_stage1_fold) == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, probs_all[:, cls])
            roc_auc_stage1[cls, fold] = auc(fpr, tpr)
            fpr_stage1[cls].append(fpr)
            tpr_stage1[cls].append(tpr)
            print(f"Stage 1 Fold {fold+1} Class {cls} AUC: {roc_auc_stage1[cls, fold]:.4f}")
        else:
            print(f"Stage 1 Fold {fold+1} Class {cls} not in test set; skipping ROC.")

    # === Stage 2 preparation using Stage 1 predictions ===
#    stage2_train_idx = [idx for i, idx in enumerate(train_idx) if y_pred_stage1[i] == 0]  # Stage 1 predicted EVENTs
    # Stage 2 training: use true labels to select EVENTs
    stage2_train_idx = [idx for idx in train_idx if y_stage1[idx] == 0]
    # Stage 2 test indices: predicted EVENTs in the current fold
    stage2_test_idx = [idx for i, idx in enumerate(test_idx) if y_pred_stage1_fold[i] == 0]
    #stage2_test_idx  = [idx for i, idx in enumerate(test_idx)  if y_pred_stage1[len(y_actual_stage1)-len(y_actual_stage1_fold)+i] == 0]

    if not stage2_train_idx or not stage2_test_idx:
        fold += 1
        continue

    # Filter out NONE class (index 2) from Stage 2
    mask_train = y_class_index[stage2_train_idx] != 2
    mask_test = y_class_index[stage2_test_idx] != 2

    X_s2_train = X[stage2_train_idx][mask_train]
    y_s2_train = y_class_index[stage2_train_idx][mask_train]
    X_s2_test = X[stage2_test_idx][mask_test]
    y_s2_test = y_class_index[stage2_test_idx][mask_test]
    filtered_test_indices = np.array(stage2_test_idx)[mask_test]

    # Remap class labels for Stage 2: ABD=0, DO=1, VOID=2
    label_map = {0: 0, 1: 1, 3: 2}
    y_s2_train = np.array([label_map[y] for y in y_s2_train])
    y_s2_test = np.array([label_map[y] for y in y_s2_test])

    # Stage 2 model
    model_s2 = NeuralNet(input_size, hidden1, hidden2, 3).to(device)
    opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=lr)
    
    # Stage 2 loss with class weights (use Stage 2 training data)
    class_weights_s2 = compute_class_weight('balanced', classes=np.unique(y_s2_train), y=y_s2_train)
    crit_s2 = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_s2, dtype=torch.float32).to(device))
    
    # Scale Stage-2 features separately
    scaler_s2 = StandardScaler().fit(X_s2_train)
    X_s2_train_scaled = scaler_s2.transform(X_s2_train)
    X_s2_test_scaled = scaler_s2.transform(X_s2_test)

    train_loader_s2 = DataLoader(UDSDataset(X_s2_train_scaled, y_s2_train), batch_size=batch_size, shuffle=True)
    test_loader_s2 = DataLoader(UDSDataset(X_s2_test_scaled, y_s2_test), batch_size=batch_size, shuffle=False)

    train_losses_s2, val_losses_s2 = [], []
    best_val_loss_s2 = float('inf')
    best_model_s2_state = None
    counter_s2 = 0

    for epoch in range(epochs):
        model_s2.train()
        epoch_loss = 0
        for xb, yb in train_loader_s2:
            xb, yb = xb.to(device), yb.to(device)
            out = model_s2(xb)
            loss = crit_s2(out, yb)
            opt_s2.zero_grad()
            loss.backward()
            opt_s2.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader_s2.dataset)
        train_losses_s2.append(epoch_loss)

        # Validation
        model_s2.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader_s2:
                xb, yb = xb.to(device), yb.to(device)
                out = model_s2(xb)
                loss = crit_s2(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(test_loader_s2.dataset)
        val_losses_s2.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss_s2:
            best_val_loss_s2 = val_loss
            best_model_s2_state = model_s2.state_dict()
            counter_s2 = 0
        else:
            counter_s2 += 1
            if counter_s2 >= patience:
                print(f"Stage 2 Early stopping at epoch {epoch+1}, fold {fold+1}")
                model_s2.load_state_dict(best_model_s2_state)
                break

    train_losses_s2_folds.append(train_losses_s2)
    val_losses_s2_folds.append(val_losses_s2)

    # Stage 2 evaluation
    model_s2.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        start_idx = 0
        for xb, yb in test_loader_s2:
            xb = xb.to(device)
            out = model_s2(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, idx in enumerate(filtered_test_indices[start_idx:start_idx+len(preds)]):
                label_map_rev = {0:'ABD',1:'DO',2:'VOID'}
                dataset.loc[idx,'class_predicted'] = label_map_rev[preds[i]]
            start_idx += len(preds)
            y_pred_stage2.extend(preds.tolist())
            y_actual_stage2.extend(yb.numpy().tolist())
            probs_all.append(probs)
            y_all.extend(yb.numpy().tolist())
            
            # --- Add this line to store Stage-2 probabilities globally ---
            y_probs_stage2_master.extend(probs.tolist())

    # Stage 2 ROC per fold
    probs_all = np.vstack(probs_all)
    y_all = np.array(y_all)
    y_bin = label_binarize(y_all, classes=[0,1,2])
    for c in range(3):
        if (y_bin[:,c].sum()>0) and (y_bin[:,c].sum()<len(y_bin)):
            fpr, tpr, _ = roc_curve(y_bin[:,c], probs_all[:,c])
            roc_auc_stage2[c, fold] = auc(fpr,tpr)
            fpr_stage2[c].append(fpr)
            tpr_stage2[c].append(tpr)

    fold += 1

# Save final models
save_dir = r'H:\Postdoc data Margot Lab\data_from_Steve\pves-classifier-main\PVES-classifier-main\PVES-classifier-main\Classification'
torch.save(model_s1.state_dict(), os.path.join(save_dir,'best_model_s1.pt'))
torch.save(model_s2.state_dict(), os.path.join(save_dir,'best_model_s2.pt'))

# === Learning curve plotting function ===
def plot_learning_curve(train_losses_folds, val_losses_folds, title, filename):
    mean_train = np.mean(train_losses_folds, axis=0)
    std_train = np.std(train_losses_folds, axis=0)
    mean_val = np.mean(val_losses_folds, axis=0)
    std_val = np.std(val_losses_folds, axis=0)
    epochs_range = np.arange(1, len(mean_train)+1)

    plt.figure()
    plt.plot(epochs_range, mean_train, label='Train Loss', color='blue')
    plt.fill_between(epochs_range, mean_train-std_train, mean_train+std_train, color='blue', alpha=0.2)
    plt.plot(epochs_range, mean_val, label='Validation Loss', color='red')
    plt.fill_between(epochs_range, mean_val-std_val, mean_val+std_val, color='red', alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
# --- Pad folds to same length ---
def pad_or_truncate(losses_list):
    max_len = max(len(l) for l in losses_list)
    padded = []
    for l in losses_list:
        if len(l) < max_len:
            l = l + [l[-1]]*(max_len - len(l))
        padded.append(l)
    return np.array(padded)

train_losses_s1_folds = pad_or_truncate(train_losses_s1_folds)
val_losses_s1_folds   = pad_or_truncate(val_losses_s1_folds)
train_losses_s2_folds = pad_or_truncate(train_losses_s2_folds)
val_losses_s2_folds   = pad_or_truncate(val_losses_s2_folds)

plot_learning_curve(train_losses_s1_folds, val_losses_s1_folds, 'Stage 1: Learning Curve', 'stage1_learning_curve.png')
plot_learning_curve(train_losses_s2_folds, val_losses_s2_folds, 'Stage 2: Learning Curve', 'stage2_learning_curve.png')

print("Training complete, models saved, and learning curves plotted.")




# Stage 1/Stage 2 evaluation, ROC, CM, etc.



# Stage 1 metrics
y_pred_stage1 = np.array(y_pred_stage1)
y_actual_stage1 = np.array(y_actual_stage1)

cm_s1 = confusion_matrix(y_actual_stage1, y_pred_stage1)
TP_s1 = cm_s1[1,1]
TN_s1 = cm_s1[0,0]
FP_s1 = cm_s1[0,1]
FN_s1 = cm_s1[1,0]

accuracy_s1 = 100 * (TP_s1 + TN_s1) / cm_s1.sum()
sensitivity_s1 = 100 * TP_s1 / (TP_s1 + FN_s1) if (TP_s1 + FN_s1) > 0 else 0
specificity_s1 = 100 * TN_s1 / (TN_s1 + FP_s1) if (TN_s1 + FP_s1) > 0 else 0
precision_s1 = 100 * TP_s1 / (TP_s1 + FP_s1) if (TP_s1 + FP_s1) > 0 else 0
f1_s1 = 2 * precision_s1 * sensitivity_s1 / (precision_s1 + sensitivity_s1) if (precision_s1 + sensitivity_s1) > 0 else 0

accuracy_per_class_s1 = 100 * np.diag(cm_s1) / cm_s1.sum(axis=1)
mean_auc_stage1 = np.mean(roc_auc_stage1, axis=1)
std_auc_stage1 = np.std(roc_auc_stage1, axis=1)

label = "Neural Network Classifier"

print(f'{label} [Stage 1: NONE vs EVENT]:')
print(f'  Accuracy = {accuracy_s1:.2f}%')
for i, cls_name in enumerate(class_labels_stage1):
    print(f'Class {cls_name} Accuracy: {accuracy_per_class_s1[i]:.2f}%')
print(f'  Sensitivity = {sensitivity_s1:.2f}%')
print(f'  Specificity = {specificity_s1:.2f}%')
print(f'  Precision = {precision_s1:.2f}%')
print(f'  F1 Score = {f1_s1:.2f}%')
print(f'Stage 1 Mean AUC per class: EVENT = {mean_auc_stage1[0]:.4f}, NONE = {mean_auc_stage1[1]:.4f}')
print(f'Stage 1 AUC StdDev per class: EVENT = {std_auc_stage1[0]:.4f}, NONE = {std_auc_stage1[1]:.4f}')
print(f'  Confusion Matrix:\n{cm_s1}\n')

# --- Stage 1 ROC ---
y_probs_stage1_master = np.array(y_probs_stage1_master)  # predicted probabilities
if y_probs_stage1_master.ndim == 2 and y_probs_stage1_master.shape[1] == 2:
    y_probs_stage1_pos = y_probs_stage1_master[:,1]
else:
    y_probs_stage1_pos = y_probs_stage1_master

min_len = min(len(y_actual_stage1), len(y_probs_stage1_pos))
y_actual_stage1 = y_actual_stage1[:min_len]
y_probs_stage1_pos = y_probs_stage1_pos[:min_len]

fpr_s1, tpr_s1, _ = roc_curve(y_actual_stage1, y_probs_stage1_pos)
roc_auc_s1 = auc(fpr_s1, tpr_s1)

plt.figure(figsize=(6,6))
plt.plot(fpr_s1, tpr_s1, color='blue', lw=2, label=f'ROC (AUC = {roc_auc_s1:.3f})')
plt.plot([0,1],[0,1],'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{label} ROC Curve (Stage 1: NONE vs EVENT)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("stage1_roc_curve_v6.png", dpi=300, bbox_inches='tight')
plt.show()

# Stage 1 confusion matrix percentages
cm_s1_percent = cm_s1.astype('float') / cm_s1.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(5,4))
sns.heatmap(cm_s1_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_labels_stage1, yticklabels=class_labels_stage1)
plt.title('Confusion Matrix (%): Stage 1: NONE vs EVENT')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("stage1_confusion_matrix_percent_v6.png", dpi=300)
plt.show()

# === Stage 2 metrics ===
# Convert to numpy arrays
y_pred_stage2 = np.array(y_pred_stage2)
y_actual_stage2 = np.array(y_actual_stage2)
y_probs_stage2_master = np.array(y_probs_stage2_master)  # predicted probabilities per class

# Confusion matrix and metrics
cm_s2 = confusion_matrix(y_actual_stage2, y_pred_stage2)
FP = cm_s2.sum(axis=0) - np.diag(cm_s2)
FN = cm_s2.sum(axis=1) - np.diag(cm_s2)
TP = np.diag(cm_s2)
TN = cm_s2.sum() - (FP + FN + TP)
sensitivity = 100 * (TP / (TP + FN))
specificity = 100 * (TN / (TN + FP))
precision = np.diag(cm_s2 / cm_s2.sum(axis=0))
recall = np.diag(cm_s2 / cm_s2.sum(axis=1))
accuracy = 100 * ((TP + TN) / (TP + FP + TN + FN))
f1 = 2 * (precision * recall) / (precision + recall)
acc_score = accuracy_score(y_actual_stage2, y_pred_stage2)

# Compute ROC and AUC per class
classes = ['ABD', 'DO', 'VOID']
n_classes = len(classes)
y_actual_binarized = label_binarize(y_actual_stage2, classes=range(n_classes))  # one-hot encoding

fpr_list = []
tpr_list = []
roc_auc_list = []

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_actual_binarized[:, i], y_probs_stage2_master[:, i])
    roc_auc = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

mean_auc = np.array(roc_auc_list)  # per class
std_auc = np.zeros_like(mean_auc)  # no folds, so std = 0

# Print Stage 2 metrics
label = "Neural Network Classifier"
print(f'{label} [Stage 2: ABD, DO, VOID]:')
print(f'  Accuracy = {acc_score*100:.2f}%')
print(f'  Accuracy (per class) = {accuracy.round(2)}%')
print(f'  Sensitivity = {sensitivity.round(2)}%')
print(f'  Specificity = {specificity.round(2)}%')
print(f'  Precision = {precision.round(2)}')
print(f'  F1 Score = {f1.round(2)}')
print(f'  Mean AUC per class = {mean_auc.round(4)}')
print(f'  AUC StdDev per class = {std_auc.round(4)}')

# Plot Stage 2 ROC curves
plt.figure(figsize=(6,6))
colors = cycle(["limegreen", "black", "cyan"])
linestyles = ['solid', 'dotted', 'dashed']
lw = 2

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_list[i], tpr_list[i], color=color, lw=lw, linestyle=linestyles[i],
             label=f'{classes[i]} (AUC = {mean_auc[i]:.3f})')

plt.plot([0,1], [0,1], "k--", lw=lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{label} ROC Curves (Stage 2)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("stage2_roc_curve_v6.png", dpi=300, bbox_inches="tight")
plt.show()

# Stage 2 confusion matrix percentages
cm_s2_percent = cm_s2.astype('float') / cm_s2.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(6,5))
sns.heatmap(cm_s2_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=['ABD','DO','VOID'], yticklabels=['ABD','DO','VOID'])
plt.title('Confusion Matrix (%): Stage 2: ABD, DO, VOID')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("stage2_confusion_matrix_percent_v6.png", dpi=300)
plt.show()

# Combined confusion matrices
fig, axes = plt.subplots(1,2,figsize=(12,5))
sns.heatmap(cm_s1_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_labels_stage1, yticklabels=class_labels_stage1, ax=axes[0])
axes[0].set_title('Stage 1 Confusion Matrix (%)')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
sns.heatmap(cm_s2_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=['ABD','DO','VOID'], yticklabels=['ABD','DO','VOID'], ax=axes[1])
axes[1].set_title('Stage 2 Confusion Matrix (%)')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
plt.tight_layout()
plt.savefig("combined_stage1_stage2_confusion_matrix_v6.png", dpi=300)
plt.show()


# Save updated CSV with predicted classes
dataset.to_csv('training_data_with_class_predicted-v6.csv', index=False)
print("Saved dataset with 'class_predicted' column.")

## Prepare dictionary to save in MATLAB format
#data_dict = {col: dataset[col].values for col in dataset.columns}

#def replace_none(obj):
#    if obj is None:
#        return np.array([])
#    elif isinstance(obj, dict):
#        return {k: replace_none(v) for k, v in obj.items()}
#    elif isinstance(obj, (list, tuple)):
#        return [replace_none(v) for v in obj]
#    elif isinstance(obj, np.ndarray) and obj.dtype == object:
#        return np.array([replace_none(v) for v in obj], dtype=object)
#    return obj

#data_dict = replace_none(data_dict)
#savemat('training_data_with_class_predicted_v6.mat', data_dict)
#print("Saved dataset in MATLAB .mat format.")

## === Generate final predictions and ground truths across all 4 classes ===

## Ground truth classes: 0=ABD, 1=DO, 2=NONE, 3=VOID
#true_labels = y_class_index  # shape: (N_samples,)

## Stage 1 predictions: 0=EVENT (ABD/DO/VOID), 1=NONE
#stage1_pred_array = np.array(y_pred_stage1)  # length = number of samples tested in Stage 1

## event_indices: global indices where Stage 2 was run (samples predicted EVENT by Stage 1)
#event_indices = np.array(stage2_pred_indices)

## Stage 2 predictions on event_indices samples: 0=ABD, 1=DO, 2=VOID
#stage2_pred_array = np.array(y_pred_stage2)

## Initialize final predictions array with all NONE (class 2)
#final_preds = np.full_like(true_labels, 2)  # default NONE

## Map Stage 2 predictions back to original labels: 0->0 (ABD), 1->1 (DO), 2->3 (VOID)
#reverse_label_map = {0: 0, 1: 1, 2: 3}
#mapped_stage2_preds = np.array([reverse_label_map[p] for p in stage2_pred_array])

## Sanity check: lengths must match
#if len(event_indices) != len(mapped_stage2_preds):
#    raise ValueError(f"Length mismatch: {len(event_indices)} event indices vs {len(mapped_stage2_preds)} Stage 2 preds")

## Assign Stage 2 predictions only at event_indices positions in final_preds
#final_preds[event_indices] = mapped_stage2_preds

## Identify all indices predicted as EVENT by Stage 1
#all_event_indices = np.where(stage1_pred_array == 0)[0]

## Find any EVENT indices missing from Stage 2 predictions (e.g., filtered out or skipped)
#missing_event_indices = np.setdiff1d(all_event_indices, event_indices)

## Assign fallback label VOID (3) for missing Stage 2 predictions at those indices
#final_preds[missing_event_indices] = 2

## Compute confusion matrix with all classes
#labels = [0, 1, 2, 3]
#class_names = ['ABD', 'DO', 'NONE', 'VOID']

#cm_combined = confusion_matrix(true_labels, final_preds, labels=labels)

## Normalize confusion matrix row-wise (true label counts)
#cm_combined_percent = cm_combined.astype(float) / cm_combined.sum(axis=1, keepdims=True) * 100

## Plot normalized confusion matrix heatmap
#plt.figure(figsize=(8,6))
#sns.heatmap(cm_combined_percent, annot=True, fmt='.1f', cmap='Blues',
#            xticklabels=class_names, yticklabels=class_names, cbar=True)

#plt.title('Confusion Matrix (%) - Combined Classifier')
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.tight_layout()
#plt.savefig("combined_confusion_matrix_percent_correct.png", dpi=300)
#plt.show()


#FP = cm_combined.sum(axis=0) - np.diag(cm_combined)
#FN = cm_combined.sum(axis=1) - np.diag(cm_combined)
#TP = np.diag(cm_combined)
#TN = cm_combined.sum() - (FP + FN + TP)

#sensitivity = 100 * (TP / (TP + FN))
#specificity = 100 * (TN / (TN + FP))
#precision = np.diag(cm_combined / cm_combined.sum(axis=0))
#recall = np.diag(cm_combined / cm_combined.sum(axis=1))
#accuracy = 100 * ((TP + TN) / (TP + FP + TN + FN))
#f1 = 2 * (precision * recall) / (precision + recall)
#acc_score = accuracy_score(true_labels, final_preds)

#print(f"Combined Classifier Accuracy = {accuracy.mean():.2f}%")
#print(f"Combined Classifier Accuracy per class = {100*np.diag(cm_combined)/cm_combined.sum(axis=1)}%")
#print(f"Sensitivity (Recall) per class = {sensitivity}")
#print(f"Specificity per class = {specificity}")
#print(f"Precision per class = {precision}")
#print(f"F1 score per class = {f1}")
#print(f"Overall accuracy_score from sklearn = {acc_score*100:.2f}%")

# ========== PVES SIGNAL INFERENCE FUNCTION ==========

def run_pves_inference_on_signal(features_file, pves_file, model_s1, model_s2, scaler, device, batch_size=200):
    """
    Run two-stage inference on a PVES signal and create visualization
    
    Parameters:
    - features_file: Path to the features CSV file
    - pves_file: Path to the PVES signal CSV file
    - model_s1: Trained Stage 1 model
    - model_s2: Trained Stage 2 model
    - scaler: Fitted StandardScaler from training
    - device: torch device
    - batch_size: Batch size for inference
    """
    
    print(f"Running inference on {features_file}...")
    
    # Load features
    X_inf = pd.read_csv(features_file)
    
    # Extract feature columns (same as training)
    features = [
        "cA1_max", "cA1_mav", "cA1_med", "cA1_ent",
        "cD1_max", "cD1_mav", "cD1_med", "cD1_ent",
        "xcorr1_med", "xcorr1_mean", "xcorr1_max",
        "cA2_max", "cA2_mav", "cA2_med", "cA2_ent",
        "cD2_max", "cD2_mav", "cD2_med", "cD2_ent",
        "xcorr2_med", "xcorr2_mean", "xcorr2_max",
        "cA3_max", "cA3_mav", "cA3_med", "cA3_ent",
        "cD3_max", "cD3_mav", "cD3_med", "cD3_ent",
        "xcorr3_med", "xcorr3_mean", "xcorr3_max",
        "cA4_max", "cA4_mav", "cA4_med", "cA4_ent",
        "cD4_max", "cD4_mav", "cD4_med", "cD4_ent",
        "xcorr4_med", "xcorr4_mean", "xcorr4_max",
        "cA5_max", "cA5_mav", "cA5_med", "cA5_ent",
        "cD5_max", "cD5_mav", "cD5_med", "cD5_ent",
        "xcorr5_med", "xcorr5_mean", "xcorr5_max"
    ]
    
    X_features = X_inf[features].to_numpy()
    
    # Load PVES signal
    pves_data = pd.read_csv(pves_file)
    P_ves = pves_data['P_ves'].to_numpy()
    
    # Load ground truth if available
    ground_truth_available = all(col in pves_data.columns for col in ['abd', 'void', 'do'])
    if ground_truth_available:
        ground_truth = pves_data[['abd', 'void', 'do']]
        
        # Check if invalid column exists
        if 'invalid' in pves_data.columns:
            invalid_annotations = pves_data['invalid'].to_numpy()
        else:
            invalid_annotations = np.zeros(len(ground_truth))
        
        # Convert to single label format
        def convert_multilabel_to_single_label(abd, void, do):
            labels = []
            for i in range(len(abd)):
                active_labels = []
                if abd[i] == 1:
                    active_labels.append('abd')
                if void[i] == 1:
                    active_labels.append('void')
                if do[i] == 1:
                    active_labels.append('do')
                
                if len(active_labels) == 0:
                    labels.append('none')
                elif len(active_labels) == 1:
                    labels.append(active_labels[0])
                else:
                    if 'abd' in active_labels:
                        labels.append('abd')
                    elif 'do' in active_labels:
                        labels.append('do')
                    elif 'void' in active_labels:
                        labels.append('void')
                    else:
                        labels.append('none')
            return labels
        
        y_ground_truth = convert_multilabel_to_single_label(
            ground_truth['abd'].to_numpy(), 
            ground_truth['void'].to_numpy(), 
            ground_truth['do'].to_numpy()
        )
        
        # Create ground truth with invalid class (4 for invalid)
        y_val = np.array([np.argmax(row) for row in label_binarize(y_ground_truth, classes=['abd', 'do', 'none', 'void'])])
        y_val[invalid_annotations == 1] = 4  # Mark invalid regions as class 4
    
    # Preprocess features
    X_features_scaled = scaler.transform(X_features)
    
    # Run two-stage inference
    print("Running Stage 1 inference (NONE vs NON-NONE)...")
    
    stage1_predictions = []
    stage1_probabilities = []
    
    model_s1.eval()
    with torch.no_grad():
        for i in range(0, len(X_features_scaled), batch_size):
            batch_X = torch.tensor(X_features_scaled[i:i+batch_size], dtype=torch.float32).to(device)
            output = model_s1(batch_X)
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            stage1_predictions.extend(preds)
            stage1_probabilities.extend(probs.cpu().numpy())
    
    stage1_predictions = np.array(stage1_predictions)
    stage1_probabilities = np.array(stage1_probabilities)
    
    print("Running Stage 2 inference (ABD, DO, VOID)...")
    
    # Find indices where Stage 1 predicted NON-NONE (class 0)
    non_none_indices = np.where(stage1_predictions == 0)[0]
    
    final_predictions = np.full(len(X_features_scaled), 2)  # Initialize all as NONE (class 2)
    stage2_probabilities = np.zeros((len(X_features_scaled), 3))  # Initialize probabilities
    
    if len(non_none_indices) > 0:
        X_stage2 = X_features_scaled[non_none_indices]
        
        model_s2.eval()
        stage2_preds = []
        stage2_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X_stage2), batch_size):
                batch_X = torch.tensor(X_stage2[i:i+batch_size], dtype=torch.float32).to(device)
                output = model_s2(batch_X)
                probs = F.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
                stage2_preds.extend(preds)
                stage2_probs.extend(probs.cpu().numpy())
        
        stage2_preds = np.array(stage2_preds)
        stage2_probs = np.array(stage2_probs)
        
        # Map Stage 2 predictions back to original indices
        final_predictions[non_none_indices] = stage2_preds
        stage2_probabilities[non_none_indices] = stage2_probs
    
    # Print statistics
    print(f"Signal length: {len(P_ves)} samples")
    print(f"Number of feature windows: {len(X_features_scaled)}")
    print(f"Window size: {len(P_ves) // len(X_features_scaled)} samples per window")
    print(f"Stage 1 - Samples classified as NONE: {np.sum(stage1_predictions == 1)}")
    print(f"Stage 1 - Samples classified as NON-NONE: {np.sum(stage1_predictions == 0)}")
    print(f"Final predictions - ABD: {np.sum(final_predictions == 0)}, DO: {np.sum(final_predictions == 1)}, NONE: {np.sum(final_predictions == 2)}")
    
    if ground_truth_available:
        print(f"Ground truth class distribution: {np.bincount(y_val)}")
        print(f"Invalid regions: {np.sum(invalid_annotations)} samples")
    
    # Create visualization
    create_pves_visualization(P_ves, final_predictions, y_val if ground_truth_available else None, 
                             features_file, ground_truth_available)
    
    return final_predictions, stage1_predictions, stage1_probabilities, stage2_probabilities

def create_pves_visualization(P_ves, final_predictions, ground_truth=None, features_file=None, ground_truth_available=False):
    """
    Create visualization of PVES signal with prediction intervals
    
    Parameters:
    - P_ves: PVES signal array
    - final_predictions: Model predictions
    - ground_truth: Ground truth labels (optional)
    - features_file: Original features file name for saving
    - ground_truth_available: Whether ground truth is available
    """
    
    # Calculate time axis (assuming 100 Hz sampling rate)
    time = np.arange(len(P_ves)) / 100.0
    
    # Calculate window size
    window_size = len(P_ves) // len(final_predictions)
    
    # Colors for different classes
    colors = ['red', 'green', 'orange', 'gray', 'purple']  # ABD, DO, NONE, VOID, INVALID
    class_names = ['ABD', 'DO', 'NONE', 'VOID', 'INVALID']
    
    # Create figure
    if ground_truth_available:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]
    
    # Plot PVES signal
    for ax in axes:
        ax.plot(time, P_ves, label='P_ves Signal', color='blue', linewidth=0.8)
        ax.set_ylabel('P_ves')
    
    # Plot ground truth if available
    if ground_truth_available:
        # Top subplot: Ground truth intervals
        intervals = []
        start_idx = 0
        current_label = ground_truth[0]
        
        for i in range(1, len(ground_truth)):
            if ground_truth[i] != current_label:
                intervals.append((start_idx, i-1, current_label))
                start_idx = i
                current_label = ground_truth[i]
        intervals.append((start_idx, len(ground_truth)-1, current_label))
        
        # Draw ground truth intervals
        y_bar = np.min(P_ves) - 0.1 * (np.max(P_ves) - np.min(P_ves))
        for start, end, label in intervals:
            if label < len(colors):
                axes[0].hlines(y=y_bar, xmin=start/100.0, xmax=(end+1)/100.0, 
                              color=colors[label], linewidth=8, 
                              label=class_names[label] if start == intervals[0][0] else "")
        
        axes[0].set_title('Ground Truth Intervals')
        handles, labels_ = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        axes[0].legend(by_label.values(), by_label.keys())
    
    # Bottom subplot: Prediction intervals
    pred_ax = axes[-1]
    
    # Create intervals for predictions
    intervals_pred = []
    start_idx_pred = 0
    current_label_pred = final_predictions[0]
    
    for i in range(1, len(final_predictions)):
        if final_predictions[i] != current_label_pred:
            # Map window indices to signal indices
            start_sample = start_idx_pred * window_size
            end_sample = (i * window_size) - 1
            intervals_pred.append((start_sample, end_sample, current_label_pred))
            start_idx_pred = i
            current_label_pred = final_predictions[i]
    
    # Add the last interval
    start_sample = start_idx_pred * window_size
    end_sample = (len(final_predictions) * window_size) - 1
    intervals_pred.append((start_sample, end_sample, current_label_pred))
    
    # Draw prediction intervals
    y_bar_pred = np.min(P_ves) - 0.1 * (np.max(P_ves) - np.min(P_ves))
    for start, end, label in intervals_pred:
        if label < len(colors):
            pred_ax.hlines(y=y_bar_pred, xmin=start/100.0, xmax=(end+1)/100.0, 
                          color=colors[label], linewidth=8, 
                          label=class_names[label] if start == intervals_pred[0][0] else "")
    
    pred_ax.set_title('Model Predictions')
    pred_ax.set_xlabel('Time (seconds)')
    
    handles_pred, labels_pred = pred_ax.get_legend_handles_labels()
    by_label_pred = dict(zip(labels_pred, handles_pred))
    pred_ax.legend(by_label_pred.values(), by_label_pred.keys())
    
    # Show color key for the classes
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], edgecolor='k', label=class_names[i]) 
                     for i in range(len(class_names))]
    fig.legend(handles=legend_patches, loc='upper center', ncol=len(class_names), 
              bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    if features_file:
        base_name = os.path.splitext(os.path.basename(features_file))[0]
        fig_filename = f"{base_name}_pves_inference.png"
    else:
        fig_filename = "pves_inference.png"
    
    fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as {fig_filename}")

def run_inference_on_multiple_signals(features_pattern, pves_pattern, model_s1, model_s2, scaler, device):
    """
    Run inference on multiple signal files matching patterns
    
    Parameters:
    - features_pattern: Glob pattern for features files (e.g., "trial_*_features.csv")
    - pves_pattern: Glob pattern for PVES files (e.g., "trial_*.csv")
    - model_s1, model_s2, scaler, device: Model and preprocessing objects
    """
    
    import glob
    
    # Get matching files
    features_files = sorted(glob.glob(features_pattern))
    pves_files = sorted(glob.glob(pves_pattern))
    
    # Filter out features files from pves_files
    pves_files = [pves_file for pves_file in pves_files 
                  if not any(features_file.replace('_features.csv', '') in pves_file 
                           for features_file in features_files)]
    
    print(f"Found {len(features_files)} features files and {len(pves_files)} PVES files")
    
    # Run inference on each pair
    for features_file, pves_file in zip(features_files, pves_files):
        print(f"\nProcessing {features_file} with {pves_file}")
        try:
            run_pves_inference_on_signal(features_file, pves_file, model_s1, model_s2, scaler, device)
        except Exception as e:
            print(f"Error processing {features_file}: {e}")
            continue

# Example usage functions
def example_single_signal_inference():
    """Example of running inference on a single signal"""
    # Load the trained models
    model_s1 = NeuralNet(input_size, hidden1, hidden2, 2).to(device)
    model_s2 = NeuralNet(input_size, hidden1, hidden2, 3).to(device)
    
    # Load saved model weights
    model_s1.load_state_dict(torch.load('best_model_s1.pt', map_location=device))
    model_s2.load_state_dict(torch.load('best_model_s2.pt', map_location=device))
    
    # Run inference on a single signal
    run_pves_inference_on_signal(
        features_file='trial_A_1_features.csv',
        pves_file='trial_A_1.csv',
        model_s1=model_s1,
        model_s2=model_s2,
        scaler=scaler,
        device=device
    )

def example_multiple_signals_inference():
    """Example of running inference on multiple signals"""
    # Load the trained models
    model_s1 = NeuralNet(input_size, hidden1, hidden2, 2).to(device)
    model_s2 = NeuralNet(input_size, hidden1, hidden2, 3).to(device)
    
    # Load saved model weights
    model_s1.load_state_dict(torch.load('best_model_s1.pt', map_location=device))
    model_s2.load_state_dict(torch.load('best_model_s2.pt', map_location=device))
    
    # Run inference on multiple signals
    run_inference_on_multiple_signals(
        features_pattern='trial_A_*_features.csv',
        pves_pattern='trial_A_*.csv',
        model_s1=model_s1,
        model_s2=model_s2,
        scaler=scaler,
        device=device
    )

# Uncomment the following lines to run inference examples:
example_single_signal_inference()
example_multiple_signals_inference()
