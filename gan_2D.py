# gan_2D.py
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.red_neuronal import inicializar_RNA
from src.activaciones import mapa_activaciones
from src.costos import mapa_costos
from src.optimizadores import mapa_optimizadores
from core.entrenamiento_gan import entrenar_gan
from core.discriminador import discriminador
from core.generador import generador

# Semilla global para reproducibilidad de los experimentos 2D
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Función para generar datos 2D reales
def generar_datos_2D(n):
    X, _ = make_moons(n_samples=n, noise=0.1)
    return X

# Parámetros
ruido_dim = 16
salida_dim = 2
tamano_lote = 128
epocas = 1500

# Generador
gen_config = {
    'input': ruido_dim,
    'capas_ocultas': [128, 64, salida_dim],
    'activaciones': ['relu', 'relu', 'lineal'],
    'lr': 0.001,
    'beta1': 0.5,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

# Discriminador
disc_config = {
    'input': salida_dim,
    'capas_ocultas': [128, 64, 1],
    'activaciones': ['relu', 'relu', 'sigmoide'],
    'lr': 0.0008,
    'beta1': 0.5,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

def init_gen():
    arquitectura = [gen_config['input']] + gen_config['capas_ocultas']
    return inicializar_RNA(arquitectura)

def init_disc():
    arquitectura = [disc_config['input']] + disc_config['capas_ocultas']
    return inicializar_RNA(arquitectura)

config = {
    'generador': gen_config,
    'discriminador': disc_config,
    'epocas': epocas,
    'tamano_lote': tamano_lote,
    'ruido_dim': ruido_dim,
    'muestras_reales': salida_dim,
    'generar_datos': lambda n: generar_datos_2D(n),
    'init_gen': init_gen,
    'init_disc': init_disc,
    'imprimir': True
}

# Entrenamiento
G, D, historial = entrenar_gan(config)

# Visualización
def mostrar_2D(G, D):
    z = np.random.normal(0, 1, (500, ruido_dim))
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        A = g(np.dot(A, W) + b)
    X_gen = A

    X_real = generar_datos_2D(500)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_real[:, 0], X_real[:, 1], color='green', label='Reales', alpha=0.6)
    plt.scatter(X_gen[:, 0], X_gen[:, 1], color='orange', label='Generados', alpha=0.6)
    plt.title("GAN 2D: Datos reales vs generados")
    plt.legend()
    plt.grid(True)
    plt.show()

mostrar_2D(G, D)

# ---------------------------
# GRÁFICAS DE EVALUACIÓN FINAL
# ---------------------------
import matplotlib.pyplot as plt

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
