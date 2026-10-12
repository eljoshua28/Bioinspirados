
from dataclasses import dataclass
import math
import random
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Sequence


@dataclass
class ResultadoPSO:
    """Contiene la mejor solución y el progreso del algoritmo."""
    mejor_posicion: list[float]
    mejor_valor: float
    historial: list[float]


def pso(
    funcion: Callable[[list[float]], float],
    limites: Sequence[tuple[float, float]],
    *,
    particulas: int = 30,
    iteraciones: int = 100,
    inercia: float = 0.7,
    aprendizaje_personal: float = 1.5,
    aprendizaje_grupal: float = 1.5,
    semilla: int | None = None,
) -> ResultadoPSO:
    """Minimiza una función; cada ejecución aleatoria puede seguir otro camino."""
    dimensiones = len(limites)
    if dimensiones == 0 or any(a >= b for a, b in limites):
        raise ValueError("Indica límites válidos: el mínimo debe ser menor al máximo.")
    if particulas < 2 or iteraciones < 1:
        raise ValueError("Usa al menos 2 partículas y 1 iteración.")
    if inercia < 0 or aprendizaje_personal < 0 or aprendizaje_grupal < 0:
        raise ValueError("Los parámetros de movimiento no pueden ser negativos.")

    azar = random.Random(semilla)

    def evaluar(posicion: list[float]) -> float:
        valor = float(funcion(posicion))
        if not math.isfinite(valor):
            raise ValueError("La función debe devolver un valor finito.")
        return valor

    # Cada partícula comienza en una posición aleatoria dentro de los límites.
    posiciones = [
        [azar.uniform(a, b) for a, b in limites]
        for _ in range(particulas)
    ]
    # La velocidad inicial pequeña permite explorar sin saltar demasiado.
    velocidades = [
        [azar.uniform(-(b - a) * 0.1, (b - a) * 0.1) for a, b in limites]
        for _ in range(particulas)
    ]

    # Cada partícula recuerda su mejor posición hasta el momento.
    mejores_personales = [p.copy() for p in posiciones]
    valores_personales = [evaluar(p) for p in posiciones]

    # El grupo también recuerda la mejor posición que cualquiera encontró.
    indice = min(range(particulas), key=lambda i: valores_personales[i])
    mejor_grupal = mejores_personales[indice].copy()
    mejor_valor = valores_personales[indice]
    historial = []

    for _ in range(iteraciones):
        for i in range(particulas):
            for d, (minimo, maximo) in enumerate(limites):
                r1, r2 = azar.random(), azar.random()

                # La velocidad mezcla movimiento anterior, memoria propia y del grupo.
                velocidades[i][d] = (
                    inercia * velocidades[i][d]
                    + aprendizaje_personal * r1 * (mejores_personales[i][d] - posiciones[i][d])
                    + aprendizaje_grupal * r2 * (mejor_grupal[d] - posiciones[i][d])
                )
                posiciones[i][d] += velocidades[i][d]

                # Mantiene la partícula dentro de los límites elegidos.
                if posiciones[i][d] < minimo:
                    posiciones[i][d] = minimo
                    velocidades[i][d] = 0.0
                elif posiciones[i][d] > maximo:
                    posiciones[i][d] = maximo
                    velocidades[i][d] = 0.0

            valor = evaluar(posiciones[i])
            if valor < valores_personales[i]:
                valores_personales[i] = valor
                mejores_personales[i] = posiciones[i].copy()
                if valor < mejor_valor:
                    mejor_valor = valor
                    mejor_grupal = posiciones[i].copy()

        historial.append(mejor_valor)

    return ResultadoPSO(mejor_grupal, mejor_valor, historial)


def esfera(valores: list[float]) -> float:
    """Función esfera: su mínimo es 0 en el punto (0, 0, ...)."""
    return sum(x * x for x in valores)


class AplicacionPSO:
    def __init__(self, ventana: tk.Tk) -> None:
        self.ventana = ventana
        ventana.title("Particle Swarm Optimization")
        ventana.minsize(620, 520)

        self.dimensiones_var = tk.StringVar(value="3")
        self.particulas_var = tk.StringVar(value="30")
        self.iteraciones_var = tk.StringVar(value="100")
        self.minimo_var = tk.StringVar(value="-10")
        self.maximo_var = tk.StringVar(value="10")
        self.resultado_var = tk.StringVar(value="Elige los parámetros y presiona «Buscar mínimo». Cada búsqueda usa azar nuevo.")

        marco = ttk.Frame(ventana, padding=16)
        marco.pack(fill="both", expand=True)
        marco.columnconfigure(1, weight=1)
        marco.rowconfigure(9, weight=1)

        ttk.Label(marco, text="Particle Swarm Optimization", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            marco,
            text="Las partículas buscan juntas el valor mínimo de una función.",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(marco, text="Función objetivo: Esfera (mínimo en el origen)").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=3
        )
        self.agregar_campo(marco, 3, "Dimensiones", self.dimensiones_var)
        self.agregar_campo(marco, 4, "Partículas", self.particulas_var)
        self.agregar_campo(marco, 5, "Iteraciones", self.iteraciones_var)
        self.agregar_campo(marco, 6, "Límite inferior", self.minimo_var)
        self.agregar_campo(marco, 7, "Límite superior", self.maximo_var)

        ttk.Button(marco, text="Buscar mínimo", command=self.ejecutar).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(10, 8)
        )
        ttk.Label(marco, textvariable=self.resultado_var, wraplength=570).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(marco, text="Progreso del mejor valor (escala logarítmica)").grid(
            row=10, column=0, columnspan=2, sticky="w"
        )
        self.grafica = tk.Canvas(marco, height=170, background="white", highlightthickness=1,
                                 highlightbackground="#cccccc")
        self.grafica.grid(row=11, column=0, columnspan=2, sticky="nsew")
        marco.rowconfigure(11, weight=1)

    @staticmethod
    def agregar_campo(marco: ttk.Frame, fila: int, etiqueta: str, variable: tk.StringVar) -> ttk.Entry:
        ttk.Label(marco, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
        entrada = ttk.Entry(marco, textvariable=variable)
        entrada.grid(row=fila, column=1, sticky="ew", pady=3)
        return entrada

    def ejecutar(self) -> None:
        try:
            dimensiones = int(self.dimensiones_var.get())
            particulas = int(self.particulas_var.get())
            iteraciones = int(self.iteraciones_var.get())
            minimo = float(self.minimo_var.get())
            maximo = float(self.maximo_var.get())
            if dimensiones < 1:
                raise ValueError("Las dimensiones deben ser al menos 1.")
            limites = [(minimo, maximo)] * dimensiones
            resultado = pso(
                esfera,
                limites,
                particulas=particulas,
                iteraciones=iteraciones,
                semilla=None,  # None genera una nueva secuencia aleatoria en cada búsqueda.
            )
            punto = ", ".join(f"{x:.8g}" for x in resultado.mejor_posicion)
            self.resultado_var.set(
                f"Mejor posición: ({punto})    |    Valor mínimo: {resultado.mejor_valor:.8g}"
            )
            self.dibujar_historial(resultado.historial)
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Revisa los parámetros", str(error))

    def dibujar_historial(self, historial: list[float]) -> None:
        # Dibujamos el progreso sin instalar bibliotecas de gráficos.
        lienzo = self.grafica
        lienzo.delete("all")
        lienzo.update_idletasks()
        ancho = max(lienzo.winfo_width(), 300)
        alto = max(lienzo.winfo_height(), 120)
        margen = 28
        datos = [math.log10(max(valor, 1e-15)) for valor in historial]
        menor, mayor = min(datos), max(datos)
        if menor == mayor:
            mayor = menor + 1.0

        lienzo.create_line(margen, margen, margen, alto - margen, fill="#666666")
        lienzo.create_line(margen, alto - margen, ancho - margen, alto - margen, fill="#666666")
        puntos = []
        for i, valor in enumerate(datos):
            x = margen + i * (ancho - 2 * margen) / max(len(datos) - 1, 1)
            y = alto - margen - (valor - menor) * (alto - 2 * margen) / (mayor - menor)
            puntos.extend((x, y))
        if len(puntos) >= 4:
            lienzo.create_line(*puntos, fill="#1769aa", width=2, smooth=True)
        lienzo.create_text(margen + 2, margen, anchor="nw", text=f"10^{mayor:.1f}", fill="#555555")
        lienzo.create_text(margen + 2, alto - margen - 16, anchor="nw", text=f"10^{menor:.1f}", fill="#555555")


def main() -> None:
    ventana = tk.Tk()
    AplicacionPSO(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
