# core/entrenamiento_gan.py
import numpy as np
import src.activaciones as act
from core.generador import generador
from core.discriminador import discriminador
import src.costos as c
import src.optimizadores as op

def entrenar_gan(config):
    gen_config = config['generador']
    disc_config = config['discriminador']
    epochs = config['epocas']
    tamano_lote = config['tamano_lote']
    ruido_dim = config['ruido_dim']
    muestras_reales = config['muestras_reales']
    generar_datos = config['generar_datos']
    imprimir = config.get('imprimir', True)

    # Inicializar pesos
    G = config['init_gen']()
    D = config['init_disc']()

    # Costos y optims
    fn_bce, dfn_bce = c.mapa_costos['bce']
    opt_gen = op.mapa_optimizadores[gen_config['optimizador']]
    opt_disc = op.mapa_optimizadores[disc_config['optimizador']]

    historial = {'gen': [], 'disc': []}

    for epoca in range(epochs):
        X_real = generar_datos(tamano_lote)
        y_real = np.ones((tamano_lote, 1)) * 0.9

        z = np.random.normal(0, 1, (tamano_lote, ruido_dim))
        X_fake = generador(z, G, gen_config['activaciones'])
        y_fake = np.zeros((tamano_lote, 1)) + 0.1

        # --- Entrenar Discriminador ---
        X_total = np.vstack([X_real, X_fake])
        y_total = np.vstack([y_real, y_fake])
        salida_d = discriminador(X_total, D, disc_config['activaciones'])
        derivadas_d = retro_d(salida_d, y_total, X_total, D, disc_config)
        D = opt_disc(D, derivadas_d, disc_config)
        loss_d = fn_bce(y_total, salida_d)

        # --- Entrenar Generador ---
        z = np.random.normal(0, 1, (tamano_lote, ruido_dim))
        X_fake = generador(z, G, gen_config['activaciones'])
        y_gen = np.ones((tamano_lote, 1))  # El generador quiere engañar
        salida_d_g = discriminador(X_fake, D, disc_config['activaciones'])
        derivadas_g = retro_g(salida_d_g, y_gen, z, G, gen_config, D, disc_config)
        G = opt_gen(G, derivadas_g, gen_config)
        loss_g = fn_bce(y_gen, salida_d_g)

        historial['gen'].append(loss_g)
        historial['disc'].append(loss_d)

        if imprimir and epoca % 10 == 0:
            print(f'Época {epoca}: Loss_D = {loss_d:.4f}, Loss_G = {loss_g:.4f}')

    return G, D, historial



# Retropropagación personalizada para D y G

def retro_d(salida, T, X, parametros, config):
    _, dfn = c.mapa_costos['bce']
    activaciones = config['activaciones']
    derivadas = {}

    # Propagación hacia adelante para guardar activaciones y derivadas
    A = X
    valores = {'a0': A}
    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, dg = act.mapa_activaciones[activacion]
        Z = np.dot(A, W) + b
        A = g(Z)
        valores[f'a{i+1}'] = A
        valores[f'da/dz{i+1}'] = dg(Z)

    # Cálculo de delta inicial
    delta = dfn(T, valores[f'a{len(activaciones)}']) * valores[f'da/dz{len(activaciones)}']

    # Retropropagación estándar
    for i in reversed(range(len(activaciones))):
        A_prev = valores[f'a{i}']
        derivadas[f'dW{i+1}'] = np.dot(A_prev.T, delta)
        derivadas[f'db{i+1}'] = np.sum(delta, axis=0, keepdims=True)
        if i != 0:
            W = parametros[f'W{i+1}']
            delta = np.dot(delta, W.T) * valores[f'da/dz{i}']

    return derivadas


def retro_g(salida_d, T, z, parametros_g, config_g, parametros_d, config_d):
    _, dfn = c.mapa_costos['bce']
    activaciones_g = config_g['activaciones']
    activaciones_d = config_d['activaciones']

    # Paso hacia adelante por G
    A = z
    valores_g = {'a0': A}
    for i, activacion in enumerate(activaciones_g):
        W = parametros_g[f'W{i+1}']
        b = parametros_g[f'b{i+1}']
        g, dg = act.mapa_activaciones[activacion]
        Z = np.dot(A, W) + b
        A = g(Z)
        valores_g[f'a{i+1}'] = A
        valores_g[f'da/dz{i+1}'] = dg(Z)

    salida_g = A

    # Paso hacia adelante por D usando salida_g
    A_d = salida_g
    for i, activacion in enumerate(activaciones_d):
        W = parametros_d[f'W{i+1}']
        b = parametros_d[f'b{i+1}']
        g, _ = act.mapa_activaciones[activacion]
        Z = np.dot(A_d, W) + b
        A_d = g(Z)

    d_loss = dfn(T, A_d)

    # Retro en G
    derivadas = {}
    delta = d_loss * valores_g[f'da/dz{len(activaciones_g)}']
    for i in reversed(range(len(activaciones_g))):
        A_prev = valores_g[f'a{i}']
        derivadas[f'dW{i+1}'] = np.dot(A_prev.T, delta)
        derivadas[f'db{i+1}'] = np.sum(delta, axis=0, keepdims=True)
        if i != 0:
            W = parametros_g[f'W{i+1}']
            delta = np.dot(delta, W.T) * valores_g[f'da/dz{i}']

    return derivadas
