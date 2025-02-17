from .eegnet import EEGNet
from .titans_model import TitansEEG
from .titans_mac_model import TitansEEGMAC
from .lstm_model import EEGLSTM

def create_model(config):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象
        
    Returns:
        nn.Module: 创建的模型实例
    """
    model_name = config['model']['name'].lower()
    
    if model_name == "eegnet":
        return EEGNet(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate'],
            kernel_length=config['model']['kernel_length'],
            num_filters=config['model']['num_filters'],
            pool_size=config['model']['pool_size']
        )
    elif model_name == "titanseeg":
        return TitansEEG(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            chunk_size=config['model']['chunk_size'],
            dropout_rate=config['model']['dropout_rate']
        )
    elif model_name == "titanseegmac":
        return TitansEEGMAC(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            chunk_size=config['model']['chunk_size'],
            num_memory_layers=config['model']['num_memory_layers'],
            dropout_rate=config['model']['dropout_rate']
        )
    elif model_name == "eeglstm":
        return EEGLSTM(
            input_channels=config['model']['input_channels'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate'],
            bidirectional=config['model']['bidirectional']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}") 