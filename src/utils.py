import numpy as np

def generar_ondas(n_muestras, rango_frec=(3.0, 3.5)):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8 * np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(*rango_frec)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2 * np.pi)
        frec_envolte = np.random.uniform(0.1, 0.3)

        envolvente = amplitud * np.sin(frec_envolte * t)
        ondas[i] = envolvente * np.sin(frecuencia * t + fase)

    return ondas

def agregar_ruido(datos_limpios, nivel=0.3):
    ruido = np.random.normal(0.0, nivel, datos_limpios.shape)
    return datos_limpios + ruido