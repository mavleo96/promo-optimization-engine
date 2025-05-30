# Promo Optimization Engine

## What's This?
A promo optimization solution we built for the **AB InBev Global Analytics Hackathon**! This repo contains our refactored version.

## Team: Simpson's Paradox
- Vijayabharathi Murugan
- Akshay Gupta
- Ashish Malhotra

## Quick Start
### What You Need
- Python 3.12+
- pip

### Get Started
```bash
pip install -r requirements.txt
```

## Project Layout
```
.
├── assets/                     # Static assets and resources
├── data_config/                # Data configuration files
├── src/                        # Source code
│   ├── synthetic_data/         # Synthetic data generation
│   ├── __init__.py
│   ├── base_module.py          # Abstract classes
│   ├── callback.py             # Training callbacks
│   ├── constants.py            # Constants
│   ├── dataset.py              # Data loading and processing
│   ├── loss.py                 # Loss functions
│   ├── model_components.py     # Model components
│   ├── model.py                # Model architecture
│   ├── opt_engine.py           # Optimization engine
│   └── utils.py                # Utility functions
├── .gitignore                  # Git ignore rules
├── main.py                     # Main training script
├── README.md                   # Project overview
├── REFERENCE.md                # This documentation
└── requirements.txt            # Project dependencies
```

## Usage

### Data Setup
Since the original data is confidential, we made a synthetic data generator to support demo and testing:

```bash
export CONFIG_PATH="data_config"
python -m src.synthetic_data --config_path $CONFIG_PATH
```

This will create synthetic data in the `data/` folder.

### Training
Run the main script with these options:

```bash
python main.py [--data_path DATA_PATH] [--epochs EPOCHS] [--opt_epochs OPT_EPOCHS] [--run_name RUN_NAME] [--num_workers NUM_WORKERS]
```

#### Parameters
- `--data_path`: Path to data directory (default: "./data")
- `--epochs`: Number of training epochs (default: 500)
- `--opt_epochs`: Number of optimization epochs (default: 500)
- `--run_name`: Name for the current run (default: "default")
- `--num_workers`: Number of data loading workers (default: 2)

## Changelog

### 2025-05-30
- Refactored to PyTorch and PyTorch Lightning
- Added synthetic data generator
- Improved documentation and project structure

### 2023-09-10
- Original solution submitted to hackathon
- Developed with TensorFlow v1
- Initial implementation of optimization engine

## License
All rights reserved by Simpson's Paradox team.