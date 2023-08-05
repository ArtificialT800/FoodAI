#Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred))*100
  return acc

#Keeping track of how long the model took to train
def time_taken(start: float,
               end: float,
               device: torch.device = device):
                 
  total_time = end-start
  return total_time


#Function to see the model's training curves
def plot_loss_curves(results):
  """Plots training curves of a trained model"""
  train_loss = results["train_loss"]
  total_test_loss = results["test_loss"]

  train_acc = results["train_acc"]
  total_test_acc = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15, 7))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss, label="Train Loss")

  plt.plot(epochs, total_test_loss, label="Test Loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_acc, label="Train Accuracy")
  plt.plot(epochs, total_test_acc, label="Test Accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


#Function to predict on custom images
def plot_preds(model: torch.nn.Module):
  """Predicts whether the image contains pizza, sushi, or steak. The probability of the predicted class will also be displayed, indicating the model's confidence in its prediction."""
  img_path = input("Enter the image path: ")
  img = torchvision.io.read_image(img_path)
  img = img.type(torch.float32) / 255
  img = img.to(device)
  img_transform = transforms.Compose([
      transforms.Resize(size=(224, 224))
  ])
  img = img_transform(img)
  img = img.unsqueeze(dim=0)

  model = model.to(device)
  model.eval()
  with torch.inference_mode():
    img_pred = model(img)
    img_pred_probs = torch.softmax(img_pred, dim=1)

  image = torch.argmax(img_pred_probs, dim=1).cpu()
  plt.figure(figsize=(12, 7))
  img = img.to("cpu")
  plot_img = img.squeeze()
  plt.imshow(plot_img.permute(1, 2, 0))
  plt.title(f"Pred: {class_names[image]} | Prob: {(img_pred_probs.max()):.4f}")
  plt.axis(False)
