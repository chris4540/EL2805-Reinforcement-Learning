# 1. Assess the impact of the model on training.
Using deeper model would be better. As the dimension of the state-space is 4 and
the dimension of the action space is 2. We use the multiple layers FNN model
with 8-16-8-4 structures.

The 8-16-8-4 gives both fast and stable convergence. Moreover, the number of
parameters is small and therefore it is a good model for this problem.