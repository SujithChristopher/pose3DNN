import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import argparse

from model import Monocular3DPoseNet, CameraAwarePoseNet, PoseLoss
from dataset_loader import create_data_loaders
from camera_utils import OV9281CameraCalibration, create_sample_camera_params

class PoseTrainer:
    """3D Pose Estimation Trainer"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = PoseLoss().to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Camera calibration
        camera_matrix, dist_coeffs = create_sample_camera_params()
        self.camera_calibration = OV9281CameraCalibration(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            keypoints_2d = batch['keypoints_2d'].to(self.device)
            keypoints_3d = batch['keypoints_3d'].to(self.device)
            
            # Create camera parameters for each sample in batch
            batch_size = keypoints_2d.shape[0]
            camera_params = self.camera_calibration.get_camera_params_tensor()
            camera_params = camera_params.unsqueeze(0).repeat(batch_size, 1).to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'lifting_net'):  # CameraAwarePoseNet
                pred_3d = self.model(keypoints_2d, camera_params)
            else:  # Monocular3DPoseNet
                pred_3d = self.model(keypoints_2d, camera_params)
            
            # Compute loss
            loss_dict = self.criterion(pred_3d, keypoints_3d)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Joint': f'{loss_dict["joint_loss"].item():.4f}',
                'Bone': f'{loss_dict["bone_loss"].item():.4f}'
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_mpjpe = 0.0  # Mean Per Joint Position Error
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                keypoints_2d = batch['keypoints_2d'].to(self.device)
                keypoints_3d = batch['keypoints_3d'].to(self.device)
                
                batch_size = keypoints_2d.shape[0]
                camera_params = self.camera_calibration.get_camera_params_tensor()
                camera_params = camera_params.unsqueeze(0).repeat(batch_size, 1).to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'lifting_net'):
                    pred_3d = self.model(keypoints_2d, camera_params)
                else:
                    pred_3d = self.model(keypoints_2d, camera_params)
                
                # Compute loss
                loss_dict = self.criterion(pred_3d, keypoints_3d)
                total_loss += loss_dict['total_loss'].item()
                
                # Compute MPJPE (mm)
                mpjpe = torch.mean(torch.norm(pred_3d - keypoints_3d, dim=2))
                total_mpjpe += mpjpe.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_mpjpe = total_mpjpe / len(val_loader)
        
        print(f"Validation Loss: {avg_loss:.4f}, MPJPE: {avg_mpjpe:.2f}mm")
        
        return avg_loss
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = 50,
              save_dir: str = './checkpoints'):
        """Main training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, train_loss, val_loss
                )
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            
            # Save latest model
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'),
                    epoch, train_loss, val_loss
                )
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filepath: str, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        ax2.plot(epochs, [lrs[0]] * len(epochs), 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train 3D Pose Estimation Model')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'camera_aware'],
                       help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(batch_size=args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'simple':
        model = Monocular3DPoseNet()
    elif args.model == 'camera_aware':
        model = CameraAwarePoseNet()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Create trainer
    trainer = PoseTrainer(model, device, learning_rate=args.lr)
    
    # Resume training if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()