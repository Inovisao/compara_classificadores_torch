import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Recall, Precision, F1Score
import glob
import argparse
import csv

parser = argparse.ArgumentParser(description='Argumentos a serem passados para determinar os processos')
parser.add_argument('-process', default=-1, help="Select the process from the following list or don't select any to run both: 0 - train only | 1 - test only")
args = parser.parse_args()
process = int(args.process)

NUM_EPOCHS = 1000
LEARNING_RATE = 0.0001
MOMENTUM = 0.5
PATIENCE_LIMIT = 10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16
BATCH_SIZE_TEST = 1
DATASET_SAMPLE_SIZE = 200
METRIC_PRECISION = 3
VAL_DATASET_PERCENTAGE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = 1
ARCHITECTURE = "BASIC_SIAMESE"
OPTIMIZER = "ADAM"
print(f'Running on {DEVICE}')

TRAIN_PATH = os.path.join('data','train')
TEST_PATH = os.path.join('data','test')
RESULTS_PATH = 'results.csv'
LOSS_GRAPH_PATH = 'loss_graphs'

def preprocess(file_path):
    img = cv2.imread(file_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    resized_img = cv2.resize(img, (105, 105))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    resized_img = resized_img.astype(np.float32) / 255.0
    transformed_img = transform(resized_img)
    return transformed_img

class SiameseDataset(Dataset):
    def __init__(self, anchor_paths, validation_paths):
        self.anchor_paths, self.validation_paths, self.labels = self.generate_balanced_pairs(anchor_paths, validation_paths)

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, index):
        anchor_img = preprocess(self.anchor_paths[index])
        validation_img = preprocess(self.validation_paths[index])
        label = self.labels[index]
        return anchor_img, validation_img, label

    def generate_balanced_pairs(self, anchor_paths, validation_paths):
        pairs_by_class = {}
        for anchor, validation in zip(anchor_paths, validation_paths):
            anchor_class = os.path.basename(os.path.dirname(anchor))
            validation_class = os.path.basename(os.path.dirname(validation))
            
            if anchor_class not in pairs_by_class:
                pairs_by_class[anchor_class] = {"positive": [], "negative": []}
            
            if anchor_class == validation_class:
                pairs_by_class[anchor_class]["positive"].append((anchor, validation, 1))
            else:
                pairs_by_class[anchor_class]["negative"].append((anchor, validation, 0))
        
        anchor_output = []
        validation_output = []
        label_output = []
        
        for class_data in pairs_by_class.values():
            positive_samples = class_data["positive"]
            negative_samples = class_data["negative"]
            
            min_samples = min(len(positive_samples), len(negative_samples))
            
            for anchor, validation, label in positive_samples[:min_samples]:
                anchor_output.append(anchor)
                validation_output.append(validation)
                label_output.append(label)
            
            for anchor, validation, label in negative_samples[:min_samples]:
                anchor_output.append(anchor)
                validation_output.append(validation)
                label_output.append(label)
        
        return anchor_output, validation_output, label_output


train_paths = [file_path for root, dirs, files in os.walk(TRAIN_PATH) for file_path in glob.glob(os.path.join(root, '*.png'))]
one_shot_data = [[preprocess(files_in_subfolder[0]), os.path.basename(sub_folder)] for sub_folder in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, sub_folder)) and (files_in_subfolder := glob.glob(os.path.join(TRAIN_PATH, sub_folder, '*.png')))]
classes = [os.path.basename(sub_folder) for sub_folder in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, sub_folder))]
num_classes = len(classes)
test_data = [[preprocess(file), os.path.basename(sub_folder)] for sub_folder in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, sub_folder)) for file in glob.glob(os.path.join(TEST_PATH, sub_folder, '*.png'))]

all_pairs = np.array(np.meshgrid(train_paths, train_paths)).T.reshape(-1, 2)
all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
anchor_paths = all_pairs[:, 0]
validation_paths = all_pairs[:, 1]
dataset = SiameseDataset(anchor_paths, validation_paths)
train_size = int((1-VAL_DATASET_PERCENTAGE) * len(dataset))
val_size = (len(dataset) - train_size)
train_data, val_data= torch.utils.data.random_split(dataset, [train_size, val_size])
steps_per_epoch = len(train_data) // BATCH_SIZE_TRAIN

train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE_VAL, shuffle=False)
print('Datasets ready for train, validation and test')

def make_embedding():
    embedding_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=10, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(64, 128, kernel_size=7, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(128, 128, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(128, 256, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(9216, 4096),
        nn.Sigmoid()
    )
    return embedding_model

class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)
    
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding = embedding_model
        self.siamese_layer = L1Dist()
        self.classifier = nn.Linear(4096, 1)
        self.model_output = nn.Sigmoid()

    def forward(self, input_image, validation_image):
        input_embedding = self.embedding(input_image)
        validation_embedding = self.embedding(validation_image)
        distances = self.siamese_layer(input_embedding, validation_embedding)
        classifier_output = self.classifier(distances)
        output = self.model_output(classifier_output)
        return output
    
print('creating siamese model')
embedding_model = make_embedding()
siamese_model = SiameseNetwork(embedding_model)
siamese_model = siamese_model.to(DEVICE)
print(f'Siamese model created with no problems: \n{siamese_model}')

if process < 1:
    print('Initializing weights and biasses')
    def initialize_weights(model):
        if isinstance(model, (nn.Conv2d, nn.Linear)):
            if isinstance(model, nn.Conv2d):
                nn.init.normal_(model.weight, mean=0, std=1e-2)
                if model.bias is not None:
                    nn.init.normal_(model.bias, mean=0.5, std=1e-2)
            elif isinstance(model, nn.Linear):
                nn.init.normal_(model.weight, mean=0, std=2e-1)
                if model.bias is not None:
                    nn.init.normal_(model.bias, mean=0.5, std=1e-2)
                    
    embedding_model.apply(initialize_weights)
    siamese_model.classifier.apply(initialize_weights)

    print('Starting train process')
    siamese_optimizer = optim.SGD(siamese_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    siamese_scheduler = optim.lr_scheduler.StepLR(siamese_optimizer, step_size=1, gamma=0.99)
    loss_fn = nn.BCELoss()
    loss_fn = loss_fn.to(DEVICE)

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    run_count = 1

    for epoch in range(NUM_EPOCHS):
        siamese_model.train()
        total_loss = 0
        
        for batch in train_data_loader:
            anchor = batch[0].to(DEVICE)
            validation = batch[1].to(DEVICE)
            labels = batch[2].float().to(DEVICE)
            outputs = siamese_model(anchor, validation)
            
            loss = loss_fn(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            siamese_optimizer.zero_grad()
            loss.backward()
            siamese_optimizer.step()
            
        average_loss = round(total_loss / steps_per_epoch, METRIC_PRECISION)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {average_loss}")
        train_losses.append(average_loss)
        siamese_scheduler.step()
        current_momentum = 0.5 + ((epoch / NUM_EPOCHS) * (MOMENTUM - 0.5))
        for param_group in siamese_optimizer.param_groups:
            param_group['momentum'] = current_momentum

        siamese_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for val_batch in val_data_loader:
                anchor = val_batch[0].to(DEVICE)
                validation = val_batch[1].to(DEVICE)
                labels = val_batch[2].float().to(DEVICE)
                outputs = siamese_model(anchor, validation)
                
                loss = loss_fn(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        average_val_loss = round(val_loss / len(val_data_loader), METRIC_PRECISION)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {average_val_loss}")
        val_losses.append(average_val_loss)

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            patience_counter = 0
            torch.save(siamese_model.state_dict(), 'siamesemodel.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_LIMIT:
                print(f"Training stopped early due to longer period with no significant reduction in evaluation metrics for the last {PATIENCE_LIMIT} epochs.")
                break

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the training losses on the first subplot
    axs[0].plot(train_losses)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')

    # Plot the validation losses on the second subplot
    axs[1].plot(val_losses)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    file_exists = True
    base_file_name = 'combined_losses.png'

    while file_exists:
        file_name = os.path.join(LOSS_GRAPH_PATH,f"{RUN}_{run_count}_{base_file_name}")
        if os.path.isfile(file_name):
            run_count += 1
        else:
            file_exists = False

    plt.savefig(file_name)

    print('Train process ended with no problems')

if process != 0:
    print('Initiating test process')
    print('Loading pre-trained weights')
    siamese_model.load_state_dict(torch.load('siamesemodel.pt'))

    precision_metric = Precision(task="multiclass", num_classes=num_classes)
    recall_metric = Recall(task="multiclass", num_classes=num_classes)
    fscore_metric = F1Score(task="multiclass", num_classes=num_classes)

    with torch.no_grad():
        for item in test_data:
            image = item[0].unsqueeze(0).to(DEVICE)
            label = next(class_id for class_id, class_name in enumerate(classes) if class_name == item[1])
            highest_score = float('-inf')
            image_class = -1
            print(len(one_shot_data))
            for anchor in one_shot_data:
                anchor_image = anchor[0].unsqueeze(0).to(DEVICE)
                pred_score = siamese_model(image, anchor_image)
                if pred_score > highest_score:
                    highest_score = pred_score
                    image_class = next(class_id for class_id, class_name in enumerate(classes) if class_name == anchor[1])
            
            print(f'Expected class: {classes[label]} Predicted class: {classes[image_class] if image_class >= 0 else "No Class Identified"} Score: {highest_score.item()}')
            
            precision_metric.update(image_class, label)
            recall_metric.update(image_class, label)
            fscore_metric.update(image_class, label)

        avg_precision = precision_metric.compute()
        avg_recall = recall_metric.compute()
        avg_fscore = fscore_metric.compute()


    print("Final results:")
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1 Score:", avg_fscore)
    print(f"Saving the results to {RESULTS_PATH}")

    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, mode='a' if file_exists else 'w', newline='') as csvfile:
        fieldnames = ["run", "learning_rate", "architecture", "optimizer", "precision", "recall", "fscore"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "run": RUN,
            "learning_rate": LEARNING_RATE,
            "architecture": ARCHITECTURE,
            "optimizer": OPTIMIZER,
            "precision": avg_precision,
            "recall": avg_recall,
            "fscore": avg_fscore
        })
    print(f"Saving process finished")

