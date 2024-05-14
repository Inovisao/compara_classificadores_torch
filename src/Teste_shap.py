import torch
import shap
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import random

data_dir = '../data/all/'

# Carregar os pesos do modelo
model_params = torch.load('../model_checkpoints/1_resnet50_adam_0.001.pth')
model = models.resnet50()
num_classes = 15
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
model.load_state_dict(model_params)

# Transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionamento para o tamanho de entrada esperado pela ResNet
    transforms.ToTensor()
])

# Carregando o dataset completo
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Defina a fração dos dados que você deseja utilizar
fraction_to_use = 0.5  # Por exemplo, para usar 50% dos dados

# Obtenha o número total de amostras no seu conjunto de dados
total_samples = len(full_dataset)

# Calcule o número de amostras que você deseja usar
num_samples_to_use = int(fraction_to_use * total_samples)

# Crie uma lista de índices aleatórios para amostrar do conjunto de dados
sampled_indices = random.sample(range(total_samples), num_samples_to_use)

# Selecionando um subconjunto aleatório do dataset completo
sampled_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)

# Crie o DataLoader para carregar os dados em lotes
batch_size = 32
print('gerando dataloader')
data_loader = torch.utils.data.DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)

# Selecionando uma amostra do DataLoader para calcular os valores SHAP
sample_input, _ = next(iter(data_loader))
print('cheguei no explainer')
# Crie o objeto Explainer com o modelo
explainer = shap.DeepExplainer(model, sample_input)
print('cheguei no shap_values')
# Calcular os valores SHAP
shap_values, indexes = explainer.shap_values(sample_input, ranked_outputs=1)
print('convertendo')
# Convertendo os tensores de valores SHAP para arrays NumPy
shap_values_numpy = [s for s in shap_values]

# Plotar os valores SHAP
#shap.image_plot(shap_values_numpy, -sample_input.numpy())

# Plotar os valores SHAP
print('iniciando ultima etapa')

for i in range(len(shap_values)):
    shap.image_plot(shap_values[i], -sample_input[i].numpy())
