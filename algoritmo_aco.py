"""Optimización por colonia de hormigas (ACO) para el problema del viajante.

Ejemplo ejecutable sin dependencias externas:
    python algoritmo_aco.py

Para usarlo con tus propios datos, llama a aco_tsp() con una matriz cuadrada
que indique la distancia entre cada par de ciudades.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


@dataclass
class ResultadoACO:
    ruta: list[int]
    distancia: float
    historial: list[float]


def aco_tsp(
    distancias: Sequence[Sequence[float]],
    *,
    hormigas: int = 30,
    iteraciones: int = 200,
    alfa: float = 1.0,
    beta: float = 3.0,
    evaporacion: float = 0.4,
    deposito: float = 1.0,
    refuerzo_mejor: float = 2.0,
    semilla: int | None = None,
) -> ResultadoACO:
    """Busca un ciclo corto con ACO y devuelve el mejor encontrado.

    ``distancias[i][j]`` es el costo de viajar de la ciudad i a la ciudad j.
    La ruta incluye cada ciudad una vez; el regreso al inicio está incluido
    al calcular ``distancia``.
    """
    n = len(distancias)
    if n < 2 or any(len(fila) != n for fila in distancias):
        raise ValueError("La matriz de distancias debe ser cuadrada y tener al menos 2 ciudades.")
    for i, fila in enumerate(distancias):
        for j, valor in enumerate(fila):
            d = float(valor)
            if not math.isfinite(d) or d < 0:
                raise ValueError("Las distancias deben ser números finitos y no negativos.")
            if i != j and d <= 0:
                raise ValueError("La distancia entre ciudades distintas debe ser mayor que cero.")
    if hormigas < 1 or iteraciones < 1:
        raise ValueError("Debe haber al menos una hormiga y una iteración.")
    if alfa < 0 or beta < 0 or not math.isfinite(alfa + beta):
        raise ValueError("alfa y beta deben ser finitos y no negativos.")
    if not 0 < evaporacion < 1:
        raise ValueError("evaporacion debe estar entre 0 y 1, sin incluir los extremos.")
    if deposito <= 0 or not math.isfinite(deposito) or refuerzo_mejor < 0:
        raise ValueError("deposito debe ser positivo y refuerzo_mejor no negativo.")

    rng = random.Random(semilla)
    feromona = [[1.0 for _ in range(n)] for _ in range(n)]
    mejor_ruta: list[int] | None = None
    mejor_distancia = math.inf
    historial: list[float] = []

    def elegir_siguiente(actual: int, disponibles: list[int]) -> int:
        # Se normalizan los logaritmos para evitar desbordamientos numéricos.
        log_pesos = [
            alfa * math.log(max(feromona[actual][ciudad], 1e-300))
            - beta * math.log(float(distancias[actual][ciudad]))
            for ciudad in disponibles
        ]
        max_log = max(log_pesos)
        pesos = [math.exp(valor - max_log) for valor in log_pesos]
        umbral = rng.random() * sum(pesos)
        acumulado = 0.0
        for ciudad, peso in zip(disponibles, pesos):
            acumulado += peso
            if acumulado >= umbral:
                return ciudad
        return disponibles[-1]

    def medir_ruta(ruta: list[int]) -> float:
        return sum(
            float(distancias[ruta[i]][ruta[(i + 1) % n]])
            for i in range(n)
        )

    for _ in range(iteraciones):
        rutas: list[list[int]] = []
        costos: list[float] = []

        for _ in range(hormigas):
            inicio = rng.randrange(n)
            ruta = [inicio]
            disponibles = [ciudad for ciudad in range(n) if ciudad != inicio]
            while disponibles:
                siguiente = elegir_siguiente(ruta[-1], disponibles)
                ruta.append(siguiente)
                disponibles.remove(siguiente)
            costo = medir_ruta(ruta)
            rutas.append(ruta)
            costos.append(costo)
            if costo < mejor_distancia:
                mejor_distancia = costo
                mejor_ruta = ruta.copy()

        historial.append(mejor_distancia)

        # Evaporación: reduce gradualmente la influencia de rutas anteriores.
        retencion = 1.0 - evaporacion
        for i in range(n):
            for j in range(n):
                feromona[i][j] *= retencion

        # Cada hormiga refuerza las aristas de su ciclo, en proporción a su calidad.
        for ruta, costo in zip(rutas, costos):
            cantidad = deposito / costo
            for i in range(n):
                a = ruta[i]
                b = ruta[(i + 1) % n]
                feromona[a][b] += cantidad

        # Refuerzo adicional para el mejor ciclo global encontrado.
        assert mejor_ruta is not None
        cantidad_mejor = refuerzo_mejor * deposito / mejor_distancia
        for i in range(n):
            a = mejor_ruta[i]
            b = mejor_ruta[(i + 1) % n]
            feromona[a][b] += cantidad_mejor

    assert mejor_ruta is not None
    return ResultadoACO(mejor_ruta, mejor_distancia, historial)


def matriz_desde_coordenadas(
    coordenadas: Sequence[tuple[float, float]],
) -> list[list[float]]:
    """Construye distancias euclidianas a partir de puntos (x, y)."""
    return [
        [
            0.0 if i == j else math.hypot(x1 - x2, y1 - y2)
            for j, (x2, y2) in enumerate(coordenadas)
        ]
        for i, (x1, y1) in enumerate(coordenadas)
    ]


def main() -> None:
    ciudades = [
        (0, 0), (2, 6), (5, 3), (6, 8),
        (8, 1), (11, 5), (12, 9), (3, 10),
    ]
    resultado = aco_tsp(
        matriz_desde_coordenadas(ciudades),
        hormigas=40,
        iteraciones=250,
        semilla=42,
    )
    ruta_cerrada = resultado.ruta + [resultado.ruta[0]]
    print("Mejor ruta:   ", " -> ".join(map(str, ruta_cerrada)))
    print("Distancia:    ", f"{resultado.distancia:.3f}")
    print("Iteraciones:  ", len(resultado.historial))


if __name__ == "__main__":
    main()
