#
#
#
#   Constantes e variáveis compartilhadas entre
#      vários arquivos
#
#
#

RASTRIGIN, ACKLEY, SPHERE, EASOM, MCCORMICK = range(5)

LOWER, HIGHER = range(2)

TOURNAMENT, FPS = range(2) # FPS = Roulette Wheel Selection or Fitness Proportionate Selection

ONE, TWO = range(2)

BOUNDS = (
    [-5.12, 5.12],
    [-5, 5],
    [-1_000, 1_000],
    [-100, 100],
    [(-1.5, 4), (-3, 4)]
)
