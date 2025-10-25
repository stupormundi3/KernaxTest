from Kernax import AbstractKernel

#On veut un hyperplane qui maximise ça More precisely, in a maximum margin method, we want to optimize the following objective
#function:
#maxw,b mini dist(xi, w, b) (2)
#such that, for all i, yi(wT φ(xi) + b) ≥ 0