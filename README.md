# Genetic Algorithm for Optimizing the Resistance of Hydrogen Tanks

## Description

This project implements a genetic algorithm in Python aimed at optimizing the resistance of hydrogen tanks used in cars pressurized to 700 bars, ensuring their ability to withstand a burst test of 1750 bars. Hydrogen tanks are critical components in various applications relying on hydrogen fuel cells, where their resistance significantly influences safety and durability.

The genetic algorithm operates by iteratively generating and evaluating populations of candidate solutions (genetic individuals). It then selects the best individuals for reproduction using genetic operators such as crossover and mutation. Each individual's fitness is determined through simulations of the tank's resistance under diverse conditions, employing a physics-based model that calculates stress in each layer of the tank.

Implemented in an object-oriented manner, the genetic algorithm encompasses classes for individual entities, populations, and the genetic algorithm itself.

## Objective

The primary goal is to discover optimal hydrogen tank designs that maximize resistance while minimizing weight and cost.

## Keywords

Genetic algorithm, optimization, hydrogen tanks, resistance, fitness function, physics-based modeling, object-oriented programming, Python, pressure, burst test, stress calculation.
