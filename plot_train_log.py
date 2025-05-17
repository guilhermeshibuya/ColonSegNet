import matplotlib.pyplot as plt

train_losses = []
val_losses = []

path = r"D:\zscan\colonsegnet\2025-05-13\train_log.txt"

with open(path, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if "Train Loss:" in lines[i]:
            train_loss = float(lines[i].split(":")[1].strip())
            val_loss = float(lines[i + 1].split(":")[1].strip())
            train_losses.append(train_loss) 
            val_losses.append(val_loss)

print(len(train_losses), len(val_losses))
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
        