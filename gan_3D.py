# gan_3D.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.red_neuronal import inicializar_RNA
from src.activaciones import mapa_activaciones
from core.entrenamiento_gan import entrenar_gan
from core.generador import generador

# Datos 3D reales: esfera con ruido
def generar_datos_3D(n):
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    r = 1 + 0.1 * np.random.randn(n)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.vstack([x, y, z]).T

# Parámetros
ruido_dim = 10
salida_dim = 3
tamano_lote = 64
epocas = 2000

gen_config = {
    'input': ruido_dim,
    'capas_ocultas': [32, 16, salida_dim],
    'activaciones': ['relu', 'relu', 'tanh'],
    'lr': 0.005,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

disc_config = {
    'input': salida_dim,
    'capas_ocultas': [32, 16, 1],
    'activaciones': ['relu', 'relu', 'sigmoide'],
    'lr': 0.005,
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
    'generar_datos': lambda n: generar_datos_3D(n),
    'init_gen': init_gen,
    'init_disc': init_disc,
    'imprimir': True
}

G, D, historial = entrenar_gan(config)

# Visualización 3D
def mostrar_3D(G):
    z = np.random.normal(0, 1, (500, ruido_dim))
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        A = g(np.dot(A, W) + b)
    X_gen = A
    X_real = generar_datos_3D(500)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_real[:, 0], X_real[:, 1], X_real[:, 2], c='green', label='Reales', alpha=0.6)
    ax.scatter(X_gen[:, 0], X_gen[:, 1], X_gen[:, 2], c='red', label='Generados', alpha=0.6)
    ax.set_title("GAN 3D: Datos reales vs generados")
    ax.legend()
    plt.tight_layout()
    plt.show()

mostrar_3D(G)

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
