# gan_1D.py
import numpy as np
import matplotlib.pyplot as plt
import src.activaciones as act
from core.entrenamiento_gan import entrenar_gan
from src.utils import generar_ondas
from src.red_neuronal import inicializar_RNA
from src.activaciones import mapa_activaciones
from src.costos import mapa_costos
from src.optimizadores import mapa_optimizadores
from core.discriminador import discriminador

# Parámetros generales
ruido_dim = 10
salida_dim = 200
tamano_lote = 64
epocas = 200

# Configuración del generador
gen_config = {
    'input': ruido_dim,
    'capas_ocultas': [64, 32, salida_dim],
    'activaciones': ['relu', 'relu', 'lineal'],
    'lr': 0.002,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

# Configuración del discriminador
disc_config = {
    'input': salida_dim,
    'capas_ocultas': [64, 32, 1],
    'activaciones': ['relu', 'relu', 'tanh'],
    'lr': 0.002,
    'optimizador': 'adam',
    'tamano_lote': tamano_lote
}

# Inicialización
def init_gen():
    arquitectura = [gen_config['input']] + gen_config['capas_ocultas']
    return inicializar_RNA(arquitectura)

def init_disc():
    arquitectura = [disc_config['input']] + disc_config['capas_ocultas']
    return inicializar_RNA(arquitectura)

# Configuración GAN
config = {
    'generador': gen_config,
    'discriminador': disc_config,
    'epocas': epocas,
    'tamano_lote': tamano_lote,
    'ruido_dim': ruido_dim,
    'muestras_reales': salida_dim,
    'generar_datos': lambda n: generar_ondas(n),
    'init_gen': init_gen,
    'init_disc': init_disc,
    'imprimir': True
}

# Entrenar
G, D, historial = entrenar_gan(config)

# 📊 Gráfica de pérdidas
def graficar_perdidas(historial):
    plt.figure(figsize=(10, 5))
    plt.plot(historial['gen'], label="Pérdida Generador", color='blue')
    plt.plot(historial['disc'], label="Pérdida Discriminador", color='red')
    plt.title("Pérdida vs Épocas")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 📈 Gráfica de ondas generadas
def mostrar_ondas_generadas(G, ruido_dim, n=5):
    z = np.random.normal(0, 1, (n, ruido_dim))
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        Z = np.dot(A, W) + b
        A = g(Z)
    ondas_generadas = A

    t = np.linspace(0, 8 * np.pi, ondas_generadas.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.plot(t, ondas_generadas[i], label=f'Onda {i+1}')
    plt.title("Ondas generadas por la GAN (1D)")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 🧠 Gráfica de confianza del discriminador
def graficar_confianza(D, G, ruido_dim):
    n = 100
    X_real = generar_ondas(n)
    z = np.random.normal(0, 1, (n, ruido_dim))

    # Propagación por G
    A = z
    for i, activacion in enumerate(gen_config['activaciones']):
        W = G[f'W{i+1}']
        b = G[f'b{i+1}']
        g, _ = mapa_activaciones[activacion]
        Z = np.dot(A, W) + b
        A = g(Z)
    X_fake = A

    # Propagación por D
    D_real = discriminador(X_real, D, disc_config['activaciones'])
    D_fake = discriminador(X_fake, D, disc_config['activaciones'])

    plt.figure(figsize=(10, 6))
    plt.hist(D_real, bins=20, alpha=0.7, label="Confianza datos REALES", color='green')
    plt.hist(D_fake, bins=20, alpha=0.7, label="Confianza datos GENERADOS", color='orange')
    plt.title("Confianza del Discriminador")
    plt.xlabel("Confianza (0 = falso, 1 = real)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejecutar visualizaciones
graficar_perdidas(historial)
mostrar_ondas_generadas(G, ruido_dim)
graficar_confianza(D, G, ruido_dim)

