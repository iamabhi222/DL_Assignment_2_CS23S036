# CS6910-Assignment2

Assignment 2 of the CS6910: Fundamentals of Deep Learning course by Abhishek Ranjan (CS23S036)

To start the training use:

- `pip install -r requirements.txt`
- `!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip`
- `!unzip -q nature_12K.zip`
- `python train_PART_A.py`

## Part A

- train_PART_A.py
- util_PART_A.py
- wandb_config_PART_A.py
- predict_PART_A.py

##### The google drive needs to be mounted and the iNaturalist file needs to be unzipped. This part of the code will need to be modified according to the filepath on your local machine.

---

```python
#Download and unzip iNaturalist zip file onto server,
!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
!unzip -q nature_12K.zip
```

There are functions defined to build a custom CNN and to prepare the image data generators for training and testing which need to be compiled.

There are 3 functions `train_wandb()`, `train()` and `predict()` for integration of WandB with the training, normal training, validation and testing process. A sweep config is defined already, whose hyperparameters and values can be modified. The train or test function can be called by the sweep agent.

```python
sweep_config = {
    "name": "Final Sweep(Bayesian)",
    "description": "Tuning hyperparameters",
    'metric': {
      'name': 'val_categorical_accuracy',
      'goal': 'maximize'
  },
    "method": "bayes",
    "parameters": {
        "n_filters": {
        "values": [16, 32]
        },
        "filter_multiplier": {
            "values": [0.5, 1, 2]
        },
        "augment_data": {
            "values": [True, False]
        },
        "dropout": {
            "values": [0.3, 0.5]
        },
        "batch_norm": {
            "values": [False, True]
        },
        "epochs": {
            "values": [5, 7, 10]
        },
        "dense_size": {
            "values": [32, 64, 128]
        },
        "lr": {
            "values": [0.01, 0.001, 0.0001]
        },
        "batch_size": {
            "values": [64, 128]
        },
        "activation": {
            "values": ["relu", "leaky"]
        },
    }
}

# creating the sweep
sweep_id = wandb.sweep(sweep_config, project="Custom", entity="-----wandb ID----")
```

The `train()` function has been made flexible with the following positional arguments which are initialized with the best parameters found by wandb sweep.

```python
def train(n_filters=32, filter_multiplier=2, dropout= 0.5,
          batch_norm = True, dense_size= 64, act_func= "relu",
          batch_size=128, augmentation=False):
```

One can also pass custom values while calling the `train()` function.

To make predictions on the model, one can run the `predict_PART_A.py` function with test data as argument.

Also, there is a function which can customise the run names in WandB.

For the code to run properly, please **modify the paths** according to your system. Following places modifications may be needed:

```python
!unzip "./nature_12K.zip"  #change path accordingly
```

`util_PART_A.py`

```python
    train_data = datasets.ImageFolder(root='./inaturalist_12K/train', transform=train_transforms) #change paths accordingly
    test_data = datasets.ImageFolder(root='./inaturalist_12K/val', transform=test_transforms) #change paths accordingly
```

`predict_PART_A.py`

```python
def predict():
    model.load_state_dict(torch.load('best_model.pth')) #change path accordingly
    return predictions
```

## Part B

- train_PART_B.py
- util_PART_B.py
- wandb_config_PART_B.py
- predict_PART_B.py

```python
def modified_model():
    # input_size = (3, 224, 224)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True

    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    return model
```

For training the model and fine-tuning with wandb, use `train()` function

```python
# Train Function
def train_wandb(config= None):
    wandb.init(project="DL-Assignment-2", entity="cs23s036")
    config = wandb.config
    wandb.run.name = setWandbName(config.epochs, config.lr, config.batch_size, config.augment_data)

    train_loader, _, val_loader = prepare_data(augmentation=config.augment_data, batch_size=config.batch_size)

    model = modified_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
```

Sweep config used:

```python
sweep_config = {
    "name": "ResNet50 Sweep(Bayesian)",
    "description": "Tuning hyperparameters",
    'metric': {
      'name': 'val_categorical_accuracy',
      'goal': 'maximize'
  },
    "method": "bayes",
    "parameters": {
        "augment_data": {
            "values": [True, False]
        },
        "epochs": {
            "values": [5, 7, 10]
        },
        "lr": {
            "values": [0.01, 0.001, 0.0001]
        },
        "batch_size": {
            "values": [16, 32]
        }
    }
}
```
