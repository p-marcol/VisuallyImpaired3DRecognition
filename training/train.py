#!/usr/bin/env python3
"""
YOLO Training Script with PyTorch
Supports YOLOv5, YOLOv8, and custom YOLO models
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """YOLO Model Trainer with checkpoint management and early stopping"""

    def __init__(self, args):
        self.args = args
        self.device = self.setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.best_fitness = 0.0
        self.patience_counter = 0

        # Create directories
        self.weights_dir = Path(args.weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = (
            self.weights_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training run directory: {self.run_dir}")
        logger.info(f"Using device: {self.device}")

    def setup_device(self) -> torch.device:
        """Setup training device (MPS, CUDA, or CPU)"""
        if self.args.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        elif self.args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration from YAML"""
        dataset_yaml = Path(self.args.data)
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

        with open(dataset_yaml, "r") as f:
            data_dict = yaml.safe_load(f)

        logger.info(f"Loaded dataset config: {dataset_yaml}")
        logger.info(f"Classes: {data_dict.get('names', [])}")
        logger.info(f"Number of classes: {data_dict.get('nc', 0)}")

        return data_dict

    def load_model(self, data_dict: Dict[str, Any]):
        """Load or create YOLO model"""
        if self.args.yolo_version == "yolov8":
            self.load_ultralytics_model(data_dict)
        elif self.args.yolo_version == "yolov5":
            self.load_yolov5_model(data_dict)
        else:
            raise ValueError(f"Unsupported YOLO version: {self.args.yolo_version}")

    def load_ultralytics_model(self, data_dict: Dict[str, Any]):
        """Load YOLOv8 using Ultralytics"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

        if self.args.weights:
            # Load existing model
            logger.info(f"Loading model from: {self.args.weights}")
            self.model = YOLO(self.args.weights)
        elif self.args.pretrained:
            # Download pretrained model
            model_name = self.args.pretrained
            logger.info(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        elif self.args.cfg:
            # Create from config
            logger.info(f"Creating model from config: {self.args.cfg}")
            self.model = YOLO(self.args.cfg)
        else:
            # Default to YOLOv8n
            logger.info("Loading default YOLOv8n model")
            self.model = YOLO("yolov8n.pt")

    def load_yolov5_model(self, data_dict: Dict[str, Any]):
        """Load YOLOv5 model"""
        try:
            import torch

            # Try to import from yolov5 package or local clone
            sys.path.append("./yolov5")
            from models.yolo import Model
            from utils.torch_utils import select_device
        except ImportError:
            raise ImportError(
                "YOLOv5 not found. Clone it: git clone https://github.com/ultralytics/yolov5"
            )

        if self.args.weights:
            logger.info(f"Loading YOLOv5 model from: {self.args.weights}")
            self.model = torch.load(self.args.weights, map_location=self.device)
        elif self.args.cfg:
            logger.info(f"Creating YOLOv5 model from config: {self.args.cfg}")
            self.model = Model(self.args.cfg, ch=3, nc=data_dict["nc"])
        else:
            raise ValueError("For YOLOv5, provide --weights or --cfg")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_fitness = checkpoint.get("best_fitness", 0.0)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Resumed from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch: int, fitness: float, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "fitness": fitness,
            "date": datetime.now().isoformat(),
        }

        # Save regular checkpoint
        if (epoch + 1) % self.args.save_period == 0:
            checkpoint_path = self.run_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.run_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Save last model
        last_path = self.run_dir / "last.pt"
        torch.save(checkpoint, last_path)

    def train_ultralytics(self, data_dict: Dict[str, Any]):
        """Train using Ultralytics YOLO"""
        # Training arguments
        train_args = {
            "data": self.args.data,
            "epochs": self.args.epochs,
            "imgsz": self.args.img_size,
            "batch": self.args.batch_size,
            "device": self.device.type,
            "project": str(self.weights_dir),
            "name": self.run_dir.name,
            "patience": self.args.patience,
            "save_period": self.args.save_period,
            "exist_ok": True,
            "pretrained": self.args.pretrained is not None,
            "optimizer": "AdamW",
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "val": True,
        }

        # Resume training
        if self.args.resume:
            train_args["resume"] = True

        logger.info("Starting training...")
        logger.info(f"Training parameters: {train_args}")

        # Train the model
        results = self.model.train(**train_args)

        logger.info("Training completed!")
        logger.info(f"Results saved to: {self.run_dir}")

        return results

    def train_yolov5(self, data_dict: Dict[str, Any]):
        """Train using YOLOv5 (requires manual implementation or using train.py from repo)"""
        logger.info(
            "For YOLOv5 training, please use the official train.py script from the repository"
        )
        logger.info(
            "Command: python yolov5/train.py --data {} --cfg {} --weights {} --batch-size {} --epochs {}".format(
                self.args.data,
                self.args.cfg or "yolov5s.yaml",
                self.args.weights or "",
                self.args.batch_size,
                self.args.epochs,
            )
        )
        raise NotImplementedError(
            "YOLOv5 training requires using the official repository's train.py"
        )

    def train(self):
        """Main training loop"""
        # Load dataset config
        data_dict = self.load_dataset_config()

        # Load model
        self.load_model(data_dict)

        # Resume from checkpoint if specified
        if self.args.resume and self.args.weights:
            self.load_checkpoint(self.args.weights)

        # Train based on YOLO version
        if self.args.yolo_version == "yolov8":
            return self.train_ultralytics(data_dict)
        elif self.args.yolo_version == "yolov5":
            return self.train_yolov5(data_dict)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLO Training Script")

    # Model selection
    parser.add_argument(
        "--yolo-version",
        type=str,
        default="yolov8",
        choices=["yolov5", "yolov8"],
        help="YOLO version to use (default: yolov8)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to existing model weights for continued training",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Path to model config file (e.g., ./models/yolo3d.yaml)",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained model to download (e.g., yolov8n.pt, yolov8s.pt, yolov5s.pt)",
    )

    # Dataset
    parser.add_argument(
        "--data", type=str, default="./dataset.yaml", help="Path to dataset YAML file"
    )

    parser.add_argument(
        "--img-size", type=int, default=640, help="Input image size (default: 640)"
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )

    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs (default: 300)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement, default: 50)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to use for training (default: mps)",
    )

    # Checkpointing
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )

    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./models/weights",
        help="Directory to save weights (default: ./models/weights)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("YOLO Training Script")
    logger.info("=" * 60)
    logger.info(f"YOLO Version: {args.yolo_version}")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Image Size: {args.img_size}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    try:
        trainer = YOLOTrainer(args)
        results = trainer.train()
        logger.info("Training finished successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
