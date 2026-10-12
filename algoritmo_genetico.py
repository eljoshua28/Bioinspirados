"""Algoritmo genético continuo, sin dependencias externas.

Ejemplo:
    python algoritmo_genetico.py

Importa genetic_algorithm() y pásale una función objetivo y los límites de cada gen.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Sequence


@dataclass
class Resultado:
    mejor_individuo: list[float]
    mejor_valor: float
    historial: list[float]
    generaciones: int


def genetic_algorithm(
    objetivo: Callable[[list[float]], float],
    limites: Sequence[tuple[float, float]],
    *,
    tamano_poblacion: int = 100,
    generaciones: int = 200,
    probabilidad_cruce: float = 0.9,
    probabilidad_mutacion: float = 0.15,
    fraccion_elite: float = 0.05,
    torneo: int = 3,
    maximizar: bool = True,
    semilla: int | None = None,
) -> Resultado:
    """Busca un óptimo mediante selección por torneo, cruce y mutación gaussiana.

    ``limites`` contiene un par (mínimo, máximo) para cada variable.
    ``objetivo`` recibe una lista de valores y devuelve un número real.
    """
    if not limites or any(a >= b for a, b in limites):
        raise ValueError("Cada límite debe tener mínimo < máximo, y debe haber al menos uno.")
    if tamano_poblacion < 2 or generaciones < 1 or torneo < 1:
        raise ValueError("Población >= 2, generaciones >= 1 y torneo >= 1.")
    if not 0 <= probabilidad_cruce <= 1 or not 0 <= probabilidad_mutacion <= 1:
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")
    if not 0 <= fraccion_elite < 1:
        raise ValueError("fraccion_elite debe estar entre 0 y 1.")

    rng = random.Random(semilla)
    dimension = len(limites)
    elite_n = min(tamano_poblacion - 1, max(1, round(tamano_poblacion * fraccion_elite)))

    def nuevo_individuo() -> list[float]:
        return [rng.uniform(minimo, maximo) for minimo, maximo in limites]

    def evaluar(individuo: list[float]) -> float:
        valor = float(objetivo(individuo))
        if not math.isfinite(valor):
            raise ValueError("La función objetivo debe devolver un número finito.")
        return valor if maximizar else -valor

    def elegir(poblacion: list[list[float]], puntuaciones: list[float]) -> list[float]:
        candidatos = [rng.randrange(len(poblacion)) for _ in range(min(torneo, len(poblacion)))]
        ganador = max(candidatos, key=lambda i: puntuaciones[i])
        return poblacion[ganador]

    poblacion = [nuevo_individuo() for _ in range(tamano_poblacion)]
    historial: list[float] = []
    mejor_global: list[float] | None = None
    mejor_puntuacion = -math.inf

    for _ in range(generaciones):
        puntuaciones = [evaluar(ind) for ind in poblacion]
        orden = sorted(range(tamano_poblacion), key=lambda i: puntuaciones[i], reverse=True)
        if puntuaciones[orden[0]] > mejor_puntuacion:
            mejor_puntuacion = puntuaciones[orden[0]]
            mejor_global = poblacion[orden[0]].copy()
        historial.append(mejor_puntuacion if maximizar else -mejor_puntuacion)

        siguiente = [poblacion[i].copy() for i in orden[:elite_n]]
        while len(siguiente) < tamano_poblacion:
            padre1 = elegir(poblacion, puntuaciones)
            padre2 = elegir(poblacion, puntuaciones)
            hijo1, hijo2 = padre1.copy(), padre2.copy()

            if rng.random() < probabilidad_cruce:
                # Cruce aritmético: cada gen combina ambos padres.
                for i in range(dimension):
                    peso = rng.random()
                    hijo1[i] = peso * padre1[i] + (1 - peso) * padre2[i]
                    hijo2[i] = peso * padre2[i] + (1 - peso) * padre1[i]

            for hijo in (hijo1, hijo2):
                for i, (minimo, maximo) in enumerate(limites):
                    if rng.random() < probabilidad_mutacion:
                        desviacion = (maximo - minimo) * 0.1
                        hijo[i] += rng.gauss(0, desviacion)
                        hijo[i] = min(max(hijo[i], minimo), maximo)
                siguiente.append(hijo)
                if len(siguiente) == tamano_poblacion:
                    break
        poblacion = siguiente

    assert mejor_global is not None
    mejor_valor = float(objetivo(mejor_global))
    return Resultado(mejor_global, mejor_valor, historial, generaciones)


def funcion_ejemplo(individuo: list[float]) -> float:
    """Máximo de f(x) = x·sin(10πx) + 1 en el intervalo [0, 1]."""
    x = individuo[0]
    return x * math.sin(10 * math.pi * x) + 1


def main() -> None:
    resultado = genetic_algorithm(
        funcion_ejemplo,
        [(0.0, 1.0)],
        semilla=42,
    )
    print("Mejor x:       ", f"{resultado.mejor_individuo[0]:.6f}")
    print("Valor máximo:  ", f"{resultado.mejor_valor:.6f}")
    print("Generaciones:  ", resultado.generaciones)


if __name__ == "__main__":
    main()
