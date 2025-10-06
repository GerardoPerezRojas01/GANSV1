import numpy as np

def gradiente_descendente(parametros, derivadas, config):
    lr = config['lr']
    parametros_actualizados = {}
    for llave in parametros.keys():
        parametros_actualizados[llave] = parametros[llave] - lr * derivadas[f'd{llave}']
    return parametros_actualizados

def gradiente_descendente_momentum(parametros, derivadas, config):
    lr = config['lr']
    beta = config.get('beta', 0.9)

    if 'velocidades' not in config:
        config['velocidades'] = {}
        for llave in parametros.keys():
            config['velocidades'][llave] = np.zeros_like(parametros[llave])

    parametros_actualizados = {}
    for llave in parametros.keys():
        config['velocidades'][llave] = beta * config['velocidades'][llave] + lr * derivadas[f'd{llave}']
        parametros_actualizados[llave] = parametros[llave] - config['velocidades'][llave]

    return parametros_actualizados

def adam(parametros, derivadas, config):
    lr = config['lr']
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    epsilon = 1e-8
    t = config.get('t', 1)

    if 'm' not in config:
        config['m'] = {k: np.zeros_like(v) for k, v in parametros.items()}
        config['v'] = {k: np.zeros_like(v) for k, v in parametros.items()}

    parametros_actualizados = {}
    for llave in parametros.keys():
        config['m'][llave] = beta1 * config['m'][llave] + (1 - beta1) * derivadas[f'd{llave}']
        config['v'][llave] = beta2 * config['v'][llave] + (1 - beta2) * (derivadas[f'd{llave}']**2)

        m_corr = config['m'][llave] / (1 - beta1**t)
        v_corr = config['v'][llave] / (1 - beta2**t)

        parametros_actualizados[llave] = parametros[llave] - lr * m_corr / (np.sqrt(v_corr) + epsilon)

    config['t'] = t + 1
    return parametros_actualizados

mapa_optimizadores = {
    'gd': gradiente_descendente,
    'gdm': gradiente_descendente_momentum,
    'adam': adam
}