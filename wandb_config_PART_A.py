import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from util_PART_A import prepare_data, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setWandbName(n_filters, filter_multiplier, augment, dropout, batch_norm):

    batch_norm_dict = {True: "Y", False: "N"}
    augment_dict = {True: "Y", False: "N"}

    name = "_".join(["num", str(n_filters), "org", str(filter_multiplier), "aug", augment_dict[augment],
                      "drop", str(dropout), "norm", batch_norm_dict[batch_norm]])

    return name


def train_wandb(config= None):
    wandb.init(project="DL-Assignment-2", entity="cs23s036")
    config = wandb.config
    wandb.run.name = setWandbName(config.n_filters, config.filter_multiplier, config.augment_data, config.dropout, config.batch_norm)

    train_loader, _, val_loader = prepare_data(augmentation=config.augment_data, batch_size=config.batch_size)

    model = CNN(n_filters=config.n_filters, filter_multiplier=config.filter_multiplier, dropout= config.dropout, batch_norm = config.batch_norm, dense_size= config.dense_size)
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
