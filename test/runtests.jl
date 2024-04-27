using SymbolicRegression

X = zeros(2, 1);
Y = zeros(1);

# equation_search(X, Y)
equation_search(X, Y; parallelism=:serial)
