import matplotlib.pyplot as plt
# gan_mnist.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.red_neuronal import inicializar_RNA
from src.activaciones import mapa_activaciones
from core.entrenamiento_gan import entrenar_gan
from core.generador import generador

# Carga MNIST desde CSV
def cargar_mnist_csv(path, n=1000):
    data = pd.read_csv(path).values[:n, 1:]  # sin etiqueta
    return data / 255.0

def generar_mnist(n):
    return cargar_mnist_csv('mnist_train.csv', n)

ruido_dim = 100
salida_dim = 784  # 28x28
tamano_lote = 64
epocas = 2000

gen_config = {
    'input': ruido_dim,
    'capas_ocultas': [128, 256, salida_dim],
    'activaciones': ['relu', 'relu', 'sigmoide'],
    'lr': 0.0002,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

disc_config = {
    'input': salida_dim,
    'capas_ocultas': [256, 128, 1],
    'activaciones': ['relu', 'relu', 'sigmoide'],
    'lr': 0.0002,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

def init_gen():
    return inicializar_RNA([ruido_dim] + gen_config['capas_ocultas'])

def init_disc():
    return inicializar_RNA([salida_dim] + disc_config['capas_ocultas'])

config = {
    'generador': gen_config,
    'discriminador': disc_config,
    'epocas': epocas,
    'tamano_lote': tamano_lote,
    'ruido_dim': ruido_dim,
    'muestras_reales': salida_dim,
    'generar_datos': lambda n: generar_mnist(n),
    'init_gen': init_gen,
    'init_disc': init_disc,
    'imprimir': True
}

G, D, historial = entrenar_gan(config)

# Visualización
def mostrar_digitos(G, ruido_dim):
    z = np.random.normal(0, 1, (10, ruido_dim))
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        A = g(np.dot(A, W) + b)

    digitos = A.reshape(-1, 28, 28)
    fig, axs = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axs[i].imshow(digitos[i], cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Dígitos generados por la GAN")
    plt.show()

mostrar_digitos(G, ruido_dim)

def graficas_loss_confianza(G, D, historial, generar_datos, ruido_dim, salida_dim):
    # ----- Gráfica de pérdidas -----
    plt.figure(figsize=(10, 5))
    plt.plot(historial['gen'], label='Pérdida Generador', color='blue')
    plt.plot(historial['disc'], label='Pérdida Discriminador', color='red')
    plt.title('Pérdidas durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----- Histograma de confianza -----
    reales = generar_datos(200)
    z = np.random.normal(0, 1, (200, ruido_dim))

    # Pasar por el generador
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        A = g(np.dot(A, W) + b)
    generadas = A

    def pasar_por_discriminador(X):
        A = X
        for i, activacion in enumerate(disc_config['activaciones']):
            W = D[f'W{i+1}']
            b = D[f'b{i+1}']
            g, _ = mapa_activaciones[activacion]
            A = g(np.dot(A, W) + b)
        return A

    conf_reales = pasar_por_discriminador(reales)
    conf_generadas = pasar_por_discriminador(generadas)

    plt.figure(figsize=(10, 5))
    plt.hist(conf_reales, bins=40, alpha=0.6, label='Confianza Reales', color='green')
    plt.hist(conf_generadas, bins=40, alpha=0.6, label='Confianza Generadas', color='orange')
    plt.title("Confianza del discriminador")
    plt.xlabel("Confianza (0=falso, 1=real)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejecuta las gráficas
graficas_loss_confianza(G, D, historial, config['generar_datos'], ruido_dim, salida_dim)
