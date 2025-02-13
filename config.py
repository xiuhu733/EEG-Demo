from typing import Dict, Any
from pathlib import Path

class Config:
    # Project paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / 'data'
    RESULTS_DIR = ROOT_DIR / 'results'
    CHECKPOINTS_DIR = ROOT_DIR / 'checkpoints'

    # Dataset configuration
    DATASET_NAME = "mkaracan/egg-quality"
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 3
    
    # Data loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Model configuration
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    
    # Training configuration
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.1
    
    # Augmentation probabilities
    AUG_HORIZONTAL_FLIP_PROB = 0.5
    AUG_BRIGHTNESS_CONTRAST_PROB = 0.2
    AUG_SHIFT_SCALE_ROTATE_PROB = 0.2
    AUG_NOISE_BLUR_PROB = 0.2
    AUG_DISTORTION_PROB = 0.2
    
    # Visualization
    NUM_SAMPLES_TO_VISUALIZE = 16
    FIGURE_SIZE = (15, 15)
    
    # Experiment tracking
    WANDB_PROJECT_NAME = "egg-classification"
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'batch_size': cls.BATCH_SIZE,
            'num_workers': cls.NUM_WORKERS,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'model_name': cls.MODEL_NAME,
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'pretrained': cls.PRETRAINED,
            'scheduler_patience': cls.SCHEDULER_PATIENCE,
            'scheduler_factor': cls.SCHEDULER_FACTOR,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'augmentation': {
                'horizontal_flip_prob': cls.AUG_HORIZONTAL_FLIP_PROB,
                'brightness_contrast_prob': cls.AUG_BRIGHTNESS_CONTRAST_PROB,
                'shift_scale_rotate_prob': cls.AUG_SHIFT_SCALE_ROTATE_PROB,
                'noise_blur_prob': cls.AUG_NOISE_BLUR_PROB,
                'distortion_prob': cls.AUG_DISTORTION_PROB,
            }
        }

# Create required directories
for directory in [Config.DATA_DIR, Config.RESULTS_DIR, Config.CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 