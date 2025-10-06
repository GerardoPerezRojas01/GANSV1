# core/generador.py
import numpy as np
import src.activaciones as act

def generador(z, parametros, activaciones):
    A = z
    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, _ = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)
    return A
