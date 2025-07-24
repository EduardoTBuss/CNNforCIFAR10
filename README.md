# 🖼️ CIFAR-10 CNN Classifier

Um classificador de imagens usando Redes Neurais Convolucionais (CNN) para o dataset CIFAR-10, implementado em PyTorch com uma arquitetura moderna e técnicas de regularização avançadas.

## 🎯 Sobre o Projeto

Este projeto implementa uma CNN profunda para classificação das 10 classes do dataset CIFAR-10:
- ✈️ Avião
- 🚗 Automóvel  
- 🐦 Pássaro
- 🐱 Gato
- 🦌 Cervo
- 🐕 Cachorro
- 🐸 Sapo
- 🐎 Cavalo
- 🚢 Navio
- 🚛 Caminhão

## 🏗️ Arquitetura da Rede

A CNN implementada possui uma arquitetura robusta com:

- **5 blocos convolucionais** com filtros crescentes (32 → 64 → 128 → 256 → 512)
- **Batch Normalization** para estabilizar o treinamento
- **Dropout** para regularização (0.25, 0.3, 0.5)
- **MaxPooling** para redução dimensional
- **Classificador totalmente conectado** com 1024 neurônios

### Técnicas de Regularização
- Data augmentation (crops aleatórios, flip horizontal, AutoAugment)
- Normalização com estatísticas do CIFAR-10
- Learning rate scheduling (decay exponencial)
- Early stopping baseado na acurácia de validação

## 📁 Estrutura do Projeto

```
CNNforCIFAR10/
├── config.py          # Configurações e hiperparâmetros
├── data.py            # Carregamento e pré-processamento dos dados
├── model.py           # Definição da arquitetura CNN
├── train.py           # Loop de treinamento e avaliação
├── plot.py            # Visualização dos resultados
├── main.py            # Script principal
└── README.md          # Este arquivo
```

## ⚙️ Configurações

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Learning Rate | 0.001 | Taxa de aprendizado inicial |
| Batch Size | 16 | Tamanho do lote |
| Gamma | 0.97 | Fator de decay do learning rate |
| Epochs | 100 | Número de épocas |
| Image Size | 32×32 | Dimensão das imagens |

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install torch torchvision matplotlib
```

### Executando o Treinamento

```bash
python main.py
```

O script irá:
1. Detectar automaticamente se GPU está disponível
2. Baixar o dataset CIFAR-10 (primeira execução)
3. Treinar o modelo com visualização em tempo real
4. Salvar o melhor modelo como `best_model.pth`
5. Gerar gráficos de treinamento em `training_plot.png`

## 📊 Saídas do Treinamento

Durante o treinamento, você verá:

```
Usando dispositivo: cuda
Epoch [1/100] | Train Loss: 1.8234 | Val Loss: 1.6543 | Val Acc: 42.15%
Epoch [2/100] | Train Loss: 1.5678 | Val Loss: 1.4321 | Val Acc: 48.72%
...
Melhor acurácia de validação: 89.45%
```

### Visualizações Geradas

- **Gráfico de Loss**: Acompanha a evolução do loss de treino e validação
- **Gráfico de Acurácia**: Monitora a acurácia de validação ao longo das épocas

## 🎯 Características Técnicas

### Data Augmentation
- **RandomCrop**: Crops aleatórios com padding
- **RandomHorizontalFlip**: Inversão horizontal aleatória  
- **AutoAugment**: Políticas automáticas de augmentação para CIFAR-10

### Otimização
- **Optimizer**: AdamW (versão melhorada do Adam)
- **Scheduler**: ExponentialLR para decay do learning rate
- **Loss Function**: CrossEntropyLoss

### Hardware
- **GPU**: Utilização automática se disponível
- **CPU**: Paralelização com múltiplos workers no DataLoader

## 📈 Resultados Esperados

Com esta arquitetura e configurações, você pode esperar:
- **Acurácia de validação**: 85-93%
- **Tempo de treinamento**: ~30-60 minutos (GPU) / 3-5 horas (CPU)
- **Convergência**: Tipicamente entre 50-80 épocas

## 🔧 Personalização

### Modificar Hiperparâmetros
Edite o arquivo `config.py`:

```python
LEARNING_RATE = 0.0005  # Reduzir para treinamento mais estável
BATCH_SIZE = 32         # Aumentar se tiver mais VRAM
EPOCHS = 150            # Mais épocas para melhor convergência
```

### Ajustar Arquitetura
Modifique o arquivo `model.py` para:
- Adicionar/remover camadas
- Alterar tamanhos de filtros
- Ajustar dropout rates

## 🤝 Contribuições

Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Implementar novas features
- Otimizar a arquitetura

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---
