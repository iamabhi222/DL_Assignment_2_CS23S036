import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from util_PART_B import modified_model
from util_PART_A import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setWandbName(epochs, lr, batch_size, augment):
    augment_dict = {True: "Y", False: "N"}

    name = "_".join(["fb", str(epochs), "aug", augment_dict[augment],
                      "drop", str(lr), "norm", "ds", str(batch_size)])

    return name


def train_wandb(config= None):
    wandb.init(project="DL-Assignment-2", entity="cs23s036")
    config = wandb.config
    wandb.run.name = setWandbName(config.epochs, config.lr, config.batch_size, config.augment_data)

    train_loader, _, val_loader = prepare_data(augmentation=config.augment_data, batch_size=config.batch_size)

    model = modified_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    for epoch in range(config.epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()   # clears the gradient

            outputs = model(inputs) # runs one forward pass
            loss = criterion(outputs, labels)   # compute the loss based on cross-entropy
            loss.backward()     # backprop
            optimizer.step()    # adam

            train_loss += loss.item()   # adds training loss for each batch

        # Log loss to WandB
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs) # runs one forward pass

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)   # compute the loss based on cross-entropy

                val_loss += loss.item()        

        val_categorical_accuracy = (100 * correct / total)
        wandb.log({"epoch": epoch + 1,
                   "Training loss": train_loss / len(train_loader),
                   "Validation loss": val_loss / len(val_loader),
                   "val_categorical_accuracy": val_categorical_accuracy})
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, val_categorical_accuracy: {val_categorical_accuracy}%")
