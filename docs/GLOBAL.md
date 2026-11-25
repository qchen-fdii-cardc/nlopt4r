# Differential evolution and particle swarm

Both functions accept borrowed `Evaluator` callbacks, validated finite `Bounds`,
typed options and a mandatory objective budget. They never request derivatives.
`value_only` provides the simplest fallible callback adapter.

```sh
cargo run --example global_search
```

## Differential evolution

`global::differential_evolution` implements DE/rand/1/bin:

1. Uniformly initialize a population (or validate a caller-supplied population).
2. For each target, select three distinct donors excluding the target.
3. Form `v=x_a+F*(x_b-x_c)`.
4. Binomial crossover selects donor coordinates with probability `CR`, with one
   forced coordinate even when `CR=0`.
5. Repair an out-of-bounds donor coordinate to the midpoint of the corresponding
   bound and target coordinate. Select the trial when its value is no worse.

All donors come from the preceding generation. This is synchronous
DE/rand/1/bin, not adaptive DE, immediate-update DE or a constrained DE method.
Defaults: 40 members, `F=0.8`, `CR=0.9`, 1000 maximum generations, seed zero,
10000 maximum objective evaluations. Valid populations have at least four
members, `0<F<=2`, `0<=CR<=1`.

## Particle swarm

`global::particle_swarm` implements inertia-weight global-best PSO:

`v_i = w*v_i + c1*r1*(pbest_i-x_i) + c2*r2*(gbest-x_i)`, then `x_i += v_i`.

Coordinate-wise independent random draws are used. Every particle in a
generation reads the same global-best snapshot. Velocities initially vanish,
and are limited to a fraction of each coordinate's domain width. Projected
coordinates have their velocities reset to zero; fixed coordinates stay fixed.
Personal bests update on strict improvement; equal values retain the earlier
best. This is not constriction PSO or local-best PSO.

Defaults: 40 particles, `w=0.7`, `c1=c2=1.5`, velocity limit `0.2`, 1000 maximum
generations, seed zero, 10000 maximum evaluations. The supported parameter
range (`0<=w<1`, nonnegative finite coefficients) is validation, not a stability
or convergence theorem.

## Reports, cancellation and limitations

`GlobalSolution` reports the best-ever evaluated point, its value, attempted
evaluation counts, completed generations, seed and normalized population
diversity. It has no fabricated gradient diagnostic. `target_reached()` means
only that the supplied target value was achieved, not certified global optimality.
Budget/iteration exhaustion and cancellation are explicit non-target reasons.

A budget may stop initialization or a generation partway through. Its best
evaluated trial is retained, and no extra callback exceeds the limit. A failure
retains value-only recovery through `Error::last_point()`.
The `_with_observer` variants expose evaluation events and fallible cancellation
callbacks; an observer failure is an error rather than a successful stop.

Storage is O(population*dimension). Positions are contiguous and candidate
buffers reused. Randomness is explicitly seeded ChaCha8 from `rand_chacha 0.9`;
record actual dependency versions/platform for reproduction.
Nonlinear constraints, parallel population evaluation, adaptive strategies and
bounded local polishing are not implemented by these entry points.

## References

- Storn and Price (1997), *Differential Evolution - A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces*,
  [doi:10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328).
- Kennedy and Eberhart (1995), *Particle swarm optimization*,
  [doi:10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968).
- Shi and Eberhart (1998), *A modified particle swarm optimizer*,
  [doi:10.1109/ICEC.1998.699146](https://doi.org/10.1109/ICEC.1998.699146).
