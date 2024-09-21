#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.dataset import Custom_stratified_Dataset
from lib.datasets.ds_tools import kfold_extract

import torch
from model.loader import model_Loader
from ptflops import get_model_complexity_info
import torch.nn as nn
import torch.optim as optim

from lib.metric.metrics import multi_classify_metrics as calculate_metrics

import wandb
from torch.cuda.amp import autocast, GradScaler

# Set environment variable for compatibility issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Disable unnecessary precision to speed up computations
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 허용 (텐서플로우 32)
torch.backends.cudnn.benchmark = True         # 최적의 알고리즘 찾기
torch.backends.cudnn.enabled = True           # cuDNN 사용
torch.autograd.set_detect_anomaly(False)

class MultiClassifier:
    def __init__(self, args):
        print('=' * 100, '=' * 100, "\033[41mStart Initialization\033[0m")
        self.args = args
        self.device = self.setup_device()
        self.wandb_use = self.args.wandb_use  # 초기화
        self.best_accuracy = 0.0  # 최적의 정확도 추적
        self.save_every = math.ceil(self.args.epochs / 10)  # 매 몇 에포크마다 저장할지 계산

        self.scaler = GradScaler()  # GradScaler 초기화

        print("\033[41mFinished Initialization\033[0m")
        self.fit()

    def setup_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\033[41mCUDA Status : {device.type}\033[0m")
        return device

    def init_kfold_dataset(self):
        kfolds, _, _ = kfold_extract(
            csv_path=self.args.csv_path, 
            n_splits=self.args.fold_num,  # 수정
            random_state=627,
            shuffle=True
        )
        return kfolds

    def init_augmentation(self):
        train_augment_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
        valid_augment_list = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]
        return train_augment_list, valid_augment_list

    def init_dataset(self, df):
        train_dataset = Custom_stratified_Dataset(
            df=df['train'], 
            root_dir=self.args.data_dir, 
            transform=transforms.Compose(self.args.train_augment_list)
        )
        val_dataset = Custom_stratified_Dataset(
            df=df['val'], 
            root_dir=self.args.data_dir, 
            transform=transforms.Compose(self.args.valid_augment_list)
        )
        
        # DataLoader 최적화
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8,           # CPU 코어 수에 맞게 조정
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.valid_batch_size,
            shuffle=False,
            num_workers=4,           # CPU 코어 수에 맞게 조정
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader

    def init_model(self, model_name, learning_rate, outlayer_num=3):
        model = model_Loader(model_name=model_name, outlayer_num=outlayer_num)()
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 수정

        if outlayer_num > 1:
            criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류
        elif outlayer_num == 1:
            criterion = nn.BCEWithLogitsLoss()  # 이진 분류
        else:
            raise ValueError(f"outlayer_num must be greater than 0.")

        print("\033[41mFinished Model Initialization\033[0m")

        return model, optimizer, criterion
    
    def calculate_model_params(self, model):
        # 모델의 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n\033[0mTotal trainable parameters: {total_params}\033[0m")

        # 모델의 FLOPs 계산
        # FLOPs 계산을 위해 입력 데이터의 형태를 정의 (예: 이미지 크기 224x224)
        input_res = (3, 224, 224)  # 예시: 3채널, 224x224 이미지
        try:
            macs, params = get_model_complexity_info(model, input_res, as_strings=False,
                                                    print_per_layer_stat=False, verbose=False)
            flops = 2 * macs  # FLOPs는 MACs의 2배
            print(f"FLOPs: {flops}")
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            flops = None
        
        self.args.total_params = total_params
        self.args.flops = flops

        
    def wandb_init(self):
        if self.wandb_use:
            print(f"\033[41mWandB Initialization\033[0m")
            wandb.init(
                project=self.args.wandb_project,
                config=self.args.__dict__
            )
            wandb.run.name = self.run_name
            wandb.watch(self.model, log="all", log_freq=100)
            print(f"\033[41mWandB Initialized\033[0m")
        else:
            print(f"\033[41mWandB Not Used\033[0m")
    
    def fit(self):
        self.kfolds = self.init_kfold_dataset()
        fold_scores = []

        for idx, fold in enumerate(self.kfolds):
            self.run_name = f"{self.args.backbone_model}_fold_{idx + 1}"
            self.args.fold_num = idx + 1
            # 데이터 증강 초기화
            self.args.train_augment_list, self.args.valid_augment_list = self.init_augmentation()
            # 데이터 로더 초기화
            self.train_loader, self.val_loader = self.init_dataset(df=fold)
            # 모델, 옵티마이저, 손실 함수 초기화
            self.model, self.optimizer, self.criterion = self.init_model(
                model_name=self.args.backbone_model, 
                outlayer_num=self.args.outlayer_num, 
                learning_rate=self.args.lr  # 수정
            )
            # 모델 파라미터 및 FLOPs 계산
            self.calculate_model_params(self.model)
            # WandB 초기화
            self.wandb_init()
            
            # Train
            print(f"\033[41mStart Training - Fold {idx + 1} \033[0m")
            val_accuracy = self.train_epoch()  # 메서드명 일치
            fold_scores.append(val_accuracy)
            
            # WandB 세션 종료
            if self.wandb_use:
                wandb.finish()

        # 모든 fold의 평균 성능 출력
        avg_score = np.mean(fold_scores)
        print(f"\n\033[42mAverage Validation Accuracy across all folds: {avg_score:.2f}%\033[0m")

    def train_epoch(self):
        final_val_accuracy = 0.0  # 최종 검증 정확도 저장
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(self.train_loader):
                # GPU로 데이터 이동 (비동기)
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                
                with autocast():  # 혼합 정밀도 사용
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                if self.args.outlayer_num > 1:
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = torch.round(torch.sigmoid(outputs))  # BCEWithLogitsLoss 사용 시
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Log training metrics to WandB every 100 steps
                if i % 100 == 0 and self.wandb_use:
                    wandb.log({
                        "train_loss": total_loss / (i + 1),
                        "train_accuracy": 100 * correct / total,
                        "epoch": epoch
                    })

            epoch_loss = total_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch [{epoch}/{self.args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            # WandB 로그
            if self.wandb_use:
                wandb.log({
                    "epoch_train_loss": epoch_loss,
                    "epoch_train_accuracy": epoch_accuracy,
                    "epoch": epoch
                })

            # Validation after each epoch
            print(f"\033[41mStart Validation - Fold {self.args.fold_num} \033[0m")
            val_accuracy = self.validate(epoch=epoch)
            final_val_accuracy = val_accuracy  # 마지막 에포크의 검증 정확도 저장
            
            # 매 save_every 에포크마다 모델 저장 및 최적 정확도 업데이트
            if epoch % self.save_every == 0:
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    save_dir = os.path.join(self.args.save_dir, self.run_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"best_model_fold_{self.args.fold_num}_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': epoch_loss,
                    }, save_path)
                    print(f"Best model saved to {save_path}")

        return final_val_accuracy  # 최종 검증 정확도 반환

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # GPU로 데이터 이동 (비동기)
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                with autocast():  # 혼합 정밀도 사용
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                if self.args.outlayer_num > 1:
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    probs = torch.sigmoid(outputs)
                    predicted = torch.round(probs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # CPU로 데이터를 이동하여 리스트에 추가
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        
        # Metrics 계산
        metrics = calculate_metrics(y_true=all_labels, y_pred=all_predictions, y_prob=np.array(all_probs), average='weighted')
        metrics['avg_loss'] = avg_loss
        metrics['epoch'] = epoch
        
        # metrics key에 'val_' prefix 추가
        metrics = {f"val_{k}": v for k, v in metrics.items()}

        # WandB 로그
        if self.wandb_use:
            wandb.log(metrics)

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {metrics['val_Accuracy']:.2f}%")
        return metrics['val_Accuracy']  # Fold별 정확도 반환

    def load_model(self, save_path):
        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 전환

        print(f"Model loaded from {save_path}, Epoch: {epoch}, Loss: {loss}")
        return checkpoint


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Multi-Classifier with WandB")
    parser.add_argument("--wandb_use", type=bool, default=False, help="WandB 실험 기록 여부")
    parser.add_argument('--wandb_project', type=str, default="multi-classifier", help='WandB project name')

    # Data Parameter
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file path')

    # Model Parameter 
    parser.add_argument('--fold_num', type=int, default=5, help='Number of folds for k-fold cross validation')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training dataset')  # 적절한 배치 사이즈로 수정
    parser.add_argument('--valid_batch_size', type=int, default=32, help='Batch size for validation dataset')  # 적절한 배치 사이즈로 수정
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # 수정
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')  # 200으로 설정
    parser.add_argument('--backbone_model', type=str, required=True, help='Backbone Model')
    parser.add_argument('--save_dir', type=str, default="saved_models", help='Directory to save models')  # 추가
    parser.add_argument('--outlayer_num', type=int, default=3, help='Number of output layers (1 for binary, >1 for multi-class)')  # 추가

    args = parser.parse_args()

    # Initialize classifier
    classifier = MultiClassifier(args)
