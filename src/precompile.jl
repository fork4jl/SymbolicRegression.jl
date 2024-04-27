using PrecompileTools: @compile_workload, @setup_workload

function do_precompilation(::Val{mode}) where {mode}
    @setup_workload begin
        X = zeros(2, 1);
        Y = zeros(1);

        @compile_workload begin
            options = SymbolicRegression.Options(;
                should_optimize_constants=false,
            )
            equation_search(
                X,
                Y;
                options=options,
                parallelism=:serial
            )
        end
    end
end
