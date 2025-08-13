import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from itertools import cycle
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
 
import math
# Dataset class
class UDSDataset(Dataset):
    def __init__(self, X_all, y_all):
        self.X = torch.tensor(X_all, dtype=torch.float32)
        self.y = torch.tensor(y_all, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

# ANN model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def convert_multilabel_to_single_label(abd, void, do):
    """Convert multi-label format to single label format"""
    labels = []
    for i in range(len(abd)):
        # Check which labels are active
        active_labels = []
        if abd[i] == 1:
            active_labels.append('abd')
        if void[i] == 1:
            active_labels.append('void')
        if do[i] == 1:
            active_labels.append('do')
        
        # Convert to single label based on priority or combination
        if len(active_labels) == 0:
            labels.append('none')
        elif len(active_labels) == 1:
            labels.append(active_labels[0])
        else:
            # Priority: abd > do > void > none
            if 'abd' in active_labels:
                labels.append('abd')
            elif 'do' in active_labels:
                labels.append('do')
            elif 'void' in active_labels:
                labels.append('void')
            else:
                labels.append('none')
    
    return labels

def load_multilabel_data(csv_file):
    """Load and preprocess multi-label dataset"""
    dataset = pd.read_csv(csv_file)
    
    # Extract multi-label columns
    abd_labels = dataset['abd'].values
    void_labels = dataset['void'].values
    do_labels = dataset['do'].values
    
    # Convert to single-label format
    y = convert_multilabel_to_single_label(abd_labels, void_labels, do_labels)
    
    # Extract feature columns (skip the first 12 columns which are label-related)
    feature_columns = dataset.columns[12:]  # Skip abd,void,do,abd_proportion,void_proportion,do_proportion,abd_count,void_count,do_count,total_labels,is_multi_label,dataset
    X = dataset[feature_columns].to_numpy()
    
    return X, y, dataset

def run_inference(X, y_stage1, y_class_index, model_s1, model_s2, device, batch_size=200):
    """Run two-stage inference on the dataset"""
    
    # Stage 1: NONE vs NON-NONE classification
    print("Running Stage 1 inference (NONE vs NON-NONE)...")
    
    stage1_predictions = []
    stage1_probabilities = []
    
    model_s1.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            output = model_s1(batch_X)
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            stage1_predictions.extend(preds)
            stage1_probabilities.extend(probs.cpu().numpy())
    
    stage1_predictions = np.array(stage1_predictions)
    stage1_probabilities = np.array(stage1_probabilities)
    
    # Stage 2: For samples predicted as NON-NONE (class 0), classify into ABD, DO, VOID
    print("Running Stage 2 inference (ABD, DO, VOID)...")
    
    # Find indices where Stage 1 predicted NON-NONE (class 0)
    non_none_indices = np.where(stage1_predictions == 0)[0]
    
    final_predictions = np.full(len(X), 2)  # Initialize all as NONE (class 2)
    stage2_probabilities = np.zeros((len(X), 3))  # Initialize probabilities
    
    if len(non_none_indices) > 0:
        X_stage2 = X[non_none_indices]
        
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
    
    return stage1_predictions, stage1_probabilities, final_predictions, stage2_probabilities

def evaluate_predictions(y_true, y_pred, stage_name):
    """Evaluate and print metrics for predictions"""
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{stage_name} Results:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, cm

def main():
    # # # ========== TRAINING SECTION ==========
    print("=== TRAINING PHASE ===")
    
    # Load training dataset
    dataset = pd.read_csv('training_data_fine.csv')
    X = dataset.iloc[:, :-1].to_numpy()
    y = dataset['class']
    y_onehot = label_binarize(y, classes=['abd', 'do', 'none', 'void'])

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare stage labels
    y_stage1 = np.array([1 if np.argmax(row) == 2 else 0 for row in y_onehot])  # 1=NONE, 0=OTHERS
    y_class_index = np.array([np.argmax(row) for row in y_onehot])  # full class index

    #Hyperparameters
    input_size = X.shape[1]
    hidden1, hidden2 = 300, 300
    batch_size = 1024
    epochs = 100
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Storage for predictions and true labels for Stage 1 and Stage 2
    y_pred_stage1 = []
    y_actual_stage1 = []
    y_pred_stage2 = []
    y_actual_stage2 = []

    roc_auc = np.zeros((3, 5))
    fpr_list = [[] for _ in range(3)]
    tpr_list = [[] for _ in range(3)]

    fold = 0
    for train_idx, test_idx in kf.split(X):
        print(f"Training fold {fold + 1}/5...")
        
        model_s1 = NeuralNet(input_size, hidden1, hidden2, 2).to(device)
        opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        train_s1 = UDSDataset(X[train_idx], y_stage1[train_idx])
        test_s1 = UDSDataset(X[test_idx], y_stage1[test_idx])
        loader_train = DataLoader(train_s1, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(test_s1, batch_size=batch_size, shuffle=False)

        for _ in range(epochs):
            for xb, yb in loader_train:
                xb, yb = xb.to(device), yb.to(device)
                out = model_s1(xb)
                loss = crit(out, yb)
                opt_s1.zero_grad()
                loss.backward()
                opt_s1.step()

        with torch.no_grad():
            for xb, yb in loader_test:
                xb = xb.to(device)
                out = model_s1(xb)
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1).cpu()
                y_pred_stage1.extend(preds.tolist())
                y_actual_stage1.extend(yb.tolist())

        # Get Stage 1 predictions for full train set
        stage2_train_indices = []
        model_s1.eval()
        with torch.no_grad():
            for i in range(0, len(train_idx), batch_size):
                idx_batch = train_idx[i:i+batch_size]
                xb = torch.tensor(X[idx_batch], dtype=torch.float32).to(device)
                out = model_s1(xb)
                preds = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()
                for j, pred in enumerate(preds):
                    if pred == 0:
                        stage2_train_indices.append(idx_batch[j])  # NON-NONE
        
        # Get Stage 1 predictions for test set
        stage2_test_indices = []
        with torch.no_grad():
            for i in range(0, len(test_idx), batch_size):
                idx_batch = test_idx[i:i+batch_size]
                xb = torch.tensor(X[idx_batch], dtype=torch.float32).to(device)
                out = model_s1(xb)
                preds = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()
                for j, pred in enumerate(preds):
                    if pred == 0:
                        stage2_test_indices.append(idx_batch[j])  # NON-NONE

        if not stage2_train_indices or not stage2_test_indices:
            fold += 1
            continue

        stage2_classes = y_class_index[stage2_test_indices]
        non_void_mask = stage2_classes != 2
        # Filter training NON-NONE/Rest (not class 2)
        train_mask = y_class_index[stage2_train_indices] != 2
        X_s2_train = X[stage2_train_indices][train_mask]
        y_s2_train = y_class_index[stage2_train_indices][train_mask]
        
        # Filter test NON-None/Rest (not class 2)
        test_mask = y_class_index[stage2_test_indices] != 2
        X_s2_test = X[stage2_test_indices][test_mask]
        y_s2_test = y_class_index[stage2_test_indices][test_mask]
        
        # Relabel ABD=0, DO=1, VOID=2 for Stage 2
        label_map = {0: 0, 1: 1, 3: 2}
        y_s2_train = np.array([label_map[y] for y in y_s2_train])
        y_s2_test = np.array([label_map[y] for y in y_s2_test])

        # Stage 2 training
        model_s2 = NeuralNet(input_size, hidden1, hidden2, 3).to(device)
        opt_s2 = torch.optim.Adam(model_s2.parameters(), lr=lr)
        train_loader_s2 = DataLoader(UDSDataset(X_s2_train, y_s2_train), batch_size=batch_size, shuffle=True)
        
        for _ in range(epochs):
            for xb, yb in train_loader_s2:
                xb, yb = xb.to(device), yb.to(device)
                out = model_s2(xb)
                loss = crit(out, yb)
                opt_s2.zero_grad()
                loss.backward()
                opt_s2.step()
        
        # Stage 2 Evaluation (on X_s2_test only!)
        test_loader_s2 = DataLoader(UDSDataset(X_s2_test, y_s2_test), batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            all_preds, all_labels, all_probs = [], [], []
            for xb, yb in test_loader_s2:
                xb = xb.to(device)
                out = model_s2(xb)
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(yb.tolist())
                all_probs.append(probs.cpu().numpy())
        
            y_pred_stage2.extend(all_preds)
            y_actual_stage2.extend(all_labels)

            all_probs_np = np.vstack(all_probs)
            y_bin = label_binarize(all_labels, classes=[0, 1, 2])
            for c in range(3):
                fpr, tpr, _ = roc_curve(y_bin[:, c], all_probs_np[:, c])
                roc_auc[c, fold] = auc(fpr, tpr)
                fpr_list[c].append(fpr)
                tpr_list[c].append(tpr)

        fold += 1

    # # Save models after training
    torch.save(model_s1.state_dict(), 'stage1_model.pt')
    torch.save(model_s2.state_dict(), 'stage2_model.pt')
    model_s1 = NeuralNet(input_size, hidden1, hidden2, 2).to(device)
    model_s2 = NeuralNet(input_size, hidden1, hidden2, 3).to(device)
    model_s1.load_state_dict(torch.load("stage1_model.pt"))
    model_s2.load_state_dict(torch.load("stage2_model.pt"));
    model_s1.eval()
    model_s2.eval()

    # # Training evaluation
    # y_pred_stage1 = np.array(y_pred_stage1)
    # y_actual_stage1 = np.array(y_actual_stage1)
    # cm_s1 = confusion_matrix(y_actual_stage1, y_pred_stage1)
    # TP_s1 = cm_s1[1,1]
    # TN_s1 = cm_s1[0,0]
    # FP_s1 = cm_s1[0,1]
    # FN_s1 = cm_s1[1,0]

    # accuracy_s1 = 100 * (TP_s1 + TN_s1) / cm_s1.sum()
    # sensitivity_s1 = 100 * TP_s1 / (TP_s1 + FN_s1) if (TP_s1 + FN_s1) > 0 else 0
    # specificity_s1 = 100 * TN_s1 / (TN_s1 + FP_s1) if (TN_s1 + FP_s1) > 0 else 0
    # precision_s1 = 100 * TP_s1 / (TP_s1 + FP_s1) if (TP_s1 + FP_s1) > 0 else 0
    # f1_s1 = 2 * precision_s1 * sensitivity_s1 / (precision_s1 + sensitivity_s1) if (precision_s1 + sensitivity_s1) > 0 else 0

    # # Stage 2 metrics
    # y_pred_stage2 = np.array(y_pred_stage2)
    # y_actual_stage2 = np.array(y_actual_stage2)
    # cm = confusion_matrix(y_actual_stage2, y_pred_stage2)
    # FP = cm.sum(axis=0) - np.diag(cm)
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm)
    # TN = cm.sum() - (FP + FN + TP)
    # sensitivity = 100 * (TP / (TP + FN))
    # specificity = 100 * (TN / (TN + FP))
    # precision = np.diag(cm / cm.sum(axis=0))
    # recall = np.diag(cm / cm.sum(axis=1))
    # accuracy = 100 * ((TP + TN) / (TP + FP + TN + FN))
    # f1 = 2 * (precision * recall) / (precision + recall)
    # acc_score = accuracy_score(y_actual_stage2, y_pred_stage2)
    # mean_auc = np.mean(roc_auc, axis=1)
    # std_auc = np.std(roc_auc, axis=1)

    # label = "Neural Network Classifier"
    # filename = "neural_classifier"

    # print(f'\n{label} [Stage 1: NONE vs NON-None]:')
    # print(f'  Accuracy = {accuracy_s1:.2f}%')
    # print(f'  Sensitivity = {sensitivity_s1:.2f}%')
    # print(f'  Specificity = {specificity_s1:.2f}%')
    # print(f'  Precision = {precision_s1:.2f}%')
    # print(f'  F1 Score = {f1_s1:.2f}%')
    # print(f'  Confusion Matrix:\n{cm_s1}\n')

    # print(f'{label} [Stage 2: ABD, DO, VOID]:')
    # print(f'  Accuracy = {acc_score*100:.2f}%')
    # print(f'  Accuracy (per class) = {accuracy.round(2)}%')
    # print(f'  Sensitivity = {sensitivity.round(2)}%')
    # print(f'  Specificity = {specificity.round(2)}%')
    # print(f'  Precision = {precision.round(2)}')
    # print(f'  F1 Score = {f1.round(2)}')
    # print(f'  Mean AUC = {mean_auc.round(4)}')
    # print(f'  AUC StdDev = {std_auc.round(4)}')

    # # Plot confusion matrix for Stage 2
    # plt.figure()
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title(f'{label} Confusion Matrix (Stage 2)')
    # plt.xlabel('Predicted Class')
    # plt.ylabel('Actual Class')
    # plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['ABD', 'DO', 'VOID'])
    # plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['ABD', 'DO', 'VOID'], rotation=0)
    # plt.savefig(filename + "_cmatrix.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # # ROC Curves
    # plt.figure()
    # colors = cycle(["limegreen", "black", "cyan"])
    # linestyles = ['solid', 'dotted', 'dashed']
    # class_names = ["ABD", "DO", "VOID"]
    # lw = 2
    # for i, color in zip(range(3), colors):
    #     for f in range(len(fpr_list[i])):
    #         plt.plot(fpr_list[i][f], tpr_list[i][f], color=color, linestyle=linestyles[i], alpha=0.3)
    #     plt.plot([], [], color=color, linestyle=linestyles[i], lw=lw,
    #              label=f'{class_names[i]} (AUC = {mean_auc[i]:.3f} ± {std_auc[i]:.3f})')

    # plt.plot([0, 1], [0, 1], "k--", lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'{label} ROC Curves (Stage 2)')
    # plt.legend(loc='lower right')
    # plt.axis('square')
    # plt.savefig(filename + "_roc.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # # ========== INFERENCE SECTION ==========
    print("\n=== INFERENCE PHASE ON DATASET A ===")
    
    # Load dataset A
    print("Loading dataset A...")
    X_inf, y_inf, dataset_raw = load_multilabel_data('training_data_multi_label_A.csv')
    
    # Preprocess inference data with same scaler as training
    X_inf = scaler.transform(X_inf)  # Use the same scaler fitted on training data
    
    # Convert to same format as training
    y_onehot_inf = label_binarize(y_inf, classes=['abd', 'do', 'none', 'void'])
    y_stage1_inf = np.array([1 if np.argmax(row) == 2 else 0 for row in y_onehot_inf])
    y_class_index_inf = np.array([np.argmax(row) for row in y_onehot_inf])
    
    print(f"Dataset A shape: {X_inf.shape}")
    print(f"Class distribution: {pd.Series(y_inf).value_counts()}")
    print(f"Multi-label statistics:")
    print(f"  ABD labels: {dataset_raw['abd'].sum()}")
    print(f"  VOID labels: {dataset_raw['void'].sum()}")
    print(f"  DO labels: {dataset_raw['do'].sum()}")
    print(f"  Multi-label samples: {dataset_raw['is_multi_label'].sum()}")
    print(f"  Total samples: {len(dataset_raw)}")
    
    # Run inference
    print("\nRunning inference...")
    stage1_preds, stage1_probs, final_preds, stage2_probs = run_inference(
        X_inf, y_stage1_inf, y_class_index_inf, model_s1, model_s2, device
    )
    
    # Evaluate Stage 1 (NONE vs NON-NONE)
    evaluate_predictions(y_stage1_inf, stage1_preds, "Stage 1 (NONE vs NON-NONE)")
    
    # Evaluate Final Predictions
    y_true_mapped = y_class_index_inf.copy()
    y_true_mapped[y_true_mapped == 3] = 2  # Map VOID (3) to 2 for final evaluation
    print(len(y_true_mapped))
    evaluate_predictions(y_true_mapped, final_preds, "Final Predictions (ABD, DO, NONE/VOID)")
    
    # Create visualizations
    # print("\nCreating visualizations...")
    
    # # Plot class distribution comparison
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(1, 2, 1)
    # class_names = ['ABD', 'DO', 'NONE', 'VOID']
    # actual_counts = [np.sum(y_class_index_inf == i) for i in range(4)]
    # plt.bar(class_names, actual_counts, alpha=0.7, color='blue')
    # plt.title('Actual Class Distribution (Dataset A)')
    # plt.ylabel('Count')
    
    # plt.subplot(1, 2, 2)
    # predicted_counts = [np.sum(final_preds == i) for i in range(3)]
    # predicted_counts.append(0)  # No VOID predictions in final stage
    # plt.bar(class_names, predicted_counts, alpha=0.7, color='red')
    # plt.title('Predicted Class Distribution (Dataset A)')
    # plt.ylabel('Count')
    
    # plt.tight_layout()
    # plt.savefig('dataset_A_class_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # Plot confusion matrix for final predictions
    # plt.figure(figsize=(8, 6))
    # cm_final = confusion_matrix(y_true_mapped, final_preds)
    # sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title('Confusion Matrix - Dataset A Inference')
    # plt.xlabel('Predicted Class')
    # plt.ylabel('Actual Class')
    # plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['ABD', 'DO', 'NONE/VOID'])
    # plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['ABD', 'DO', 'NONE/VOID'], rotation=0)
    # plt.savefig('dataset_A_confusion_matrix.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'actual_class': y_inf,
        'stage1_prediction': ['NONE' if p == 1 else 'NON-NONE' for p in stage1_preds],
        'stage1_confidence': np.max(stage1_probs, axis=1),
        'final_prediction': ['ABD' if p == 0 else 'DO' if p == 1 else 'NONE' for p in final_preds],
        'final_confidence': np.max(stage2_probs, axis=1)
    })
    
    results_df.to_csv('dataset_A_predictions.csv', index=False)
    print("\nPredictions saved to 'dataset_A_predictions.csv'")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {len(X_inf)}")
    print(f"Stage 1 - Samples classified as NONE: {np.sum(stage1_preds == 1)}")
    print(f"Stage 1 - Samples classified as NON-NONE: {np.sum(stage1_preds == 0)}")
    print(f"Stage 2 - Samples processed: {np.sum(stage1_preds == 0)}")
    print(f"Final - ABD predictions: {np.sum(final_preds == 0)}")
    print(f"Final - DO predictions: {np.sum(final_preds == 1)}")
    print(f"Final - NONE predictions: {np.sum(final_preds == 2)}")


    



    #run inference on each trial_A_[num]_features.csv file with overlay over pves signal
    files = sorted(glob.glob("trial_A_*_features.csv"))
    pvesFiles = sorted(glob.glob("trial_A_*.csv"));

    #if file in files and pvesFiles remove it from pvesFiles
    pvesFiles = [pvesFile for pvesFile in pvesFiles if not any(file in pvesFile for file in files)]
    print(pvesFiles,files)
    plt.ion() # Start interactive mode
    for (file, pvesFile) in zip(files, pvesFiles):

        X_inf = pd.read_csv(file)

        # Skip the first 12 columns which are label-related and exclude non-numeric columns
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

        # features = features.iloc[:, 17:]  # Skip the first 12 columns which are label-related
        
        # Remove non-numeric columns (like 'dataset', 'trial_id', 'window_id', etc.)
        #numeric_columns = features.select_dtypes(include=[np.number]).columns
        features = X_inf[features]
  
        P_ves = pd.read_csv(pvesFile)['P_ves'].to_numpy()  # Load P_ves signal
        
        # Load ground truth and invalid annotations
        pves_data = pd.read_csv(pvesFile)
        ground_truth = pves_data[['abd', 'void', 'do']]
        
        # Check if invalid column exists
        if 'invalid' in pves_data.columns:
            invalid_annotations = pves_data['invalid'].to_numpy()
        else:
            invalid_annotations = np.zeros(len(ground_truth))
        
        # Convert to single label format, but mark invalid regions
        y_ground_truth = convert_multilabel_to_single_label(
            ground_truth['abd'].to_numpy(), 
            ground_truth['void'].to_numpy(), 
            ground_truth['do'].to_numpy()
        )
        
        # Create ground truth with invalid class (4 for invalid)
        y_val = np.array([np.argmax(row) for row in label_binarize(y_ground_truth, classes=['abd', 'do', 'none', 'void'])])
        
        # Mark invalid regions as class 4 (invalid)
        y_val[invalid_annotations == 1] = 4




        X_inf = features.to_numpy()
        print(f"Running inference on {file}...")   
        # Preprocess inference data with same scaler as training
        X_inf = scaler.transform(X_inf)
        # Run inference
        stage1_preds, stage1_probs, final_preds, stage2_probs = run_inference(
            X_inf, y_stage1_inf, y_class_index_inf, model_s1, model_s2, device
        )
        #get P_ves signal for corresponding trial

        print(f"Difference in inference size: {len(X_inf)*8} vs {len(P_ves)}. Difference is {len(P_ves) - len(X_inf)*8}")
        print(f"Ground truth class distribution: {np.bincount(y_val)}")
        print(f"Invalid regions: {np.sum(invalid_annotations)} samples")
        print(y_val)


        #Adjust time axis by dividing by 100
        time = np.arange(len(P_ves)) / 100.0

        #Create two plots:
        #Plot 1: P_ves signal with highlighted intervals of the ground truth
        #Plot 2: P_ves signal with highlighted intervals of the final predictions

        #Plot 1
        # Plot 1
        # Create a single figure with two subplots: top for ground truth, bottom for predictions
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top subplot: Ground truth intervals
        axes[0].plot(time, P_ves, label='P_ves Signal', color='blue')
        axes[0].set_ylabel('P_ves')
        colors = ['red', 'green', 'orange', 'gray', 'purple']  # ABD, DO, NONE, VOID, INVALID
        class_names = ['abd', 'do', 'none', 'void', 'invalid']
        intervals = []
        start_idx = 0
        current_label = y_val[0]
        for i in range(1, len(y_val)):
            if y_val[i] != current_label:
                intervals.append((start_idx, i-1, current_label))
                start_idx = i
                current_label = y_val[i]
        intervals.append((start_idx, len(y_val)-1, current_label))
        # Draw intervals as horizontal bars below the signal
        y_bar = np.min(P_ves) - 0.1 * (np.max(P_ves) - np.min(P_ves))
        bar_height = 0.05 * (np.max(P_ves) - np.min(P_ves))
        for start, end, label in intervals:
            if label < len(colors):  # Ensure label is within valid range
                axes[0].hlines(y=y_bar, xmin=start/100.0, xmax=(end+1)/100.0, color=colors[label], linewidth=8, label=class_names[label] if start == intervals[0][0] else "")
        handles, labels_ = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        axes[0].legend(by_label.values(), by_label.keys())

        # Bottom subplot: Final prediction intervals
        axes[1].plot(time, P_ves, label='P_ves Signal', color='blue')
        axes[1].set_xlabel('Time (arbitrary units)')
        axes[1].set_ylabel('P_ves')
        
        # Calculate window size and total signal length
        window_size = 8  # Each prediction corresponds to 8 samples
        total_windows = len(final_preds)
        expected_signal_length = total_windows * window_size
        
        # Create intervals for predictions that span the full signal
        intervals_pred = []
        start_idx_pred = 0
        current_label_pred = final_preds[0]
        
        for i in range(1, len(final_preds)):
            if final_preds[i] != current_label_pred:
                # Map window indices to signal indices
                start_sample = start_idx_pred * window_size
                end_sample = (i * window_size) - 1
                intervals_pred.append((start_sample, end_sample, current_label_pred))
                start_idx_pred = i
                current_label_pred = final_preds[i]
        
        # Add the last interval
        start_sample = start_idx_pred * window_size
        end_sample = (len(final_preds) * window_size) - 1
        intervals_pred.append((start_sample, end_sample, current_label_pred))
        
        y_bar_pred = np.min(P_ves) - 0.1 * (np.max(P_ves) - np.min(P_ves))
        for start, end, label in intervals_pred:
            if label < len(colors):  # Ensure label is within valid range
                axes[1].hlines(y=y_bar_pred, xmin=start/100.0, xmax=(end+1)/100.0, color=colors[label], linewidth=8, label=class_names[label] if start == intervals_pred[0][0] else "")
        
        handles_pred, labels_pred = axes[1].get_legend_handles_labels()
        by_label_pred = dict(zip(labels_pred, handles_pred))
        axes[1].legend(by_label_pred.values(), by_label_pred.keys())

        # Show color key for the classes above the plots
        legend_patches = [Patch(facecolor=colors[i], edgecolor='k', label=class_names[i]) for i in range(len(class_names))]
        fig.legend(handles=legend_patches, loc='upper center', ncol=len(class_names), bbox_to_anchor=(0.5, 1.04))

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig_filename = f"{os.path.splitext(os.path.basename(file))[0]}_intervals.png"
        fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.show(block=False)

    plt.ioff() # Turn off interactive mode





    




    


if __name__ == "__main__":
    main()