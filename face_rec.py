from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#número de classes (pessoas que você está tentando reconhecer)
num_classes = 10  # Por exemplo

# carrega o modelo MobileNetV2 pré-treinado no ImageNet, excluindo a camada de saída
base_model = MobileNetV2(weights='imagenet', include_top=False)

# nova camada de saída para classificar as pessoas
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# definir o novo modelo com a nova camada de saída
model = Model(inputs=base_model.input, outputs=predictions)

#pesos do modelo base (não treiná-los novamente)
for layer in base_model.layers:
    layer.trainable = False

# compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# dir. imagens de treinamento divididas em subdiretórios por classe
train_data_dir = '/caminho/para/seus/dados/de/treinamento'

#  tamanho das imagens de entrada
img_width, img_height = 224, 224

#gerador de imagens de treinamento com aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Redimensione os valores dos pixels para estar no intervalo [0, 1]
    shear_range=0.2,   # Aplicar cisalhamento aleatório
    zoom_range=0.2,    # Aplicar zoom aleatório
    horizontal_flip=True)  # Inverter aleatoriamente as imagens horizontalmente

#lotes de dados de treinamento usando o gerador
train_generator = train_datagen.flow_from_directory(
    train_data_dir,        # Diretório das imagens de treinamento
    target_size=(img_width, img_height),  # Redimensione as imagens para o tamanho esperado do modelo
    batch_size=32,         # Tamanho do lote
    class_mode='categorical')  # Modo de classificação para rótulos categóricos

# número total de amostras de treinamento
total_train = train_generator.samples

# treinamento do modelo com dados específicos de reconhecimento facial
model.fit_generator(train_generator, 
                    steps_per_epoch=total_train // 32,  # Número de etapas por época
                    epochs=10)  # Número de épocas


