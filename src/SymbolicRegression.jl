module SymbolicRegression

export equation_search


using Distributed
using Printf: @printf, @sprintf
using Pkg: Pkg
using TOML: parsefile
using Random: seed!, shuffle!
using DynamicExpressions:
    Node,
    GraphNode,
    NodeSampler,
    AbstractExpressionNode,
    copy_node,
    set_node!,
    string_tree,
    print_tree,
    count_nodes,
    get_constants,
    set_constants,
    index_constants,
    NodeIndex,
    eval_tree_array,
    differentiable_eval_tree_array,
    eval_diff_tree_array,
    eval_grad_tree_array,
    node_to_symbolic,
    symbolic_to_node,
    combine_operators,
    simplify_tree!,
    tree_mapreduce,
    set_default_variable_names!
using DynamicExpressions.EquationModule: with_type_parameters
using LossFunctions: SupervisedLoss


# https://discourse.julialang.org/t/how-to-find-out-the-version-of-a-package-from-its-module/37755/15
const PACKAGE_VERSION = try
    root = pkgdir(@__MODULE__)
    if root == String
        let project = parsefile(joinpath(root, "Project.toml"))
            VersionNumber(project["version"])
        end
    else
        VersionNumber(0, 0, 0)
    end
catch
    VersionNumber(0, 0, 0)
end

function deprecate_varmap(variable_names, varMap, func_name)
    if varMap !== nothing
        Base.depwarn("`varMap` is deprecated; use `variable_names` instead", func_name)
        @assert variable_names === nothing "Cannot pass both `varMap` and `variable_names`"
        variable_names = varMap
    end
    return variable_names
end

include("Utils.jl")
include("InterfaceDynamicQuantities.jl")
include("Core.jl")
include("InterfaceDynamicExpressions.jl")
include("Recorder.jl")
include("Complexity.jl")
include("DimensionalAnalysis.jl")
include("CheckConstraints.jl")
include("AdaptiveParsimony.jl")
include("MutationFunctions.jl")
include("LossFunctions.jl")
include("PopMember.jl")
include("ConstantOptimization.jl")
include("Population.jl")
include("HallOfFame.jl")
include("Mutate.jl")
include("RegularizedEvolution.jl")
include("SingleIteration.jl")
include("ProgressBars.jl")
include("Migration.jl")
include("SearchUtils.jl")

using .CoreModule:
    MAX_DEGREE,
    BATCH_DIM,
    FEATURE_DIM,
    DATA_TYPE,
    LOSS_TYPE,
    RecordType,
    Dataset,
    Options,
    MutationWeights,
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    cond,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip
using .UtilsModule: is_anonymous_function, recursive_merge, json3_write
using .ComplexityModule: compute_complexity
using .CheckConstraintsModule: check_constraints
using .AdaptiveParsimonyModule:
    RunningSearchStatistics, update_frequencies!, move_window!, normalize_frequencies!
using .MutationFunctionsModule:
    gen_random_tree,
    gen_random_tree_fixed_size,
    random_node,
    random_node_and_parent,
    crossover_trees
using .InterfaceDynamicExpressionsModule: @extend_operators
using .LossFunctionsModule: eval_loss, score_func, update_baseline_loss!
using .PopMemberModule: PopMember, reset_birth!
using .PopulationModule: Population, best_sub_pop, record_population, best_of_sample
using .HallOfFameModule:
    HallOfFame, calculate_pareto_frontier, string_dominating_pareto_curve
using .SingleIterationModule: s_r_cycle, optimize_and_simplify_population
using .ProgressBarsModule: WrappedProgressBar
using .RecorderModule: @recorder, find_iteration_from_record
using .MigrationModule: migrate!
using .SearchUtilsModule:
    SearchState,
    RuntimeOptions,
    WorkerAssignments,
    DefaultWorkerOutputType,
    assign_next_worker!,
    get_worker_output_type,
    extract_from_worker,
    @sr_spawner,
    StdinReader,
    watch_stream,
    close_reader!,
    check_for_user_quit,
    check_for_loss_threshold,
    check_for_timeout,
    check_max_evals,
    ResourceMonitor,
    start_work_monitor!,
    stop_work_monitor!,
    estimate_work_fraction,
    update_progress_bar!,
    print_search_state,
    init_dummy_pops,
    load_saved_hall_of_fame,
    load_saved_population,
    construct_datasets,
    get_cur_maxsize,
    update_hall_of_fame!

include("Configure.jl")


function equation_search(
    X::AbstractMatrix{T},
    y::AbstractMatrix{T};
    niterations::Int=10,
    weights::Union{AbstractMatrix{T},AbstractVector{T},Nothing}=nothing,
    options::Options=Options(),
    variable_names::Union{AbstractVector{String},Nothing}=nothing,
    display_variable_names::Union{AbstractVector{String},Nothing}=variable_names,
    y_variable_names::Union{String,AbstractVector{String},Nothing}=nothing,
    parallelism=:multithreading,
    numprocs::Union{Int,Nothing}=nothing,
    procs::Union{Vector{Int},Nothing}=nothing,
    addprocs_function::Union{Function,Nothing}=nothing,
    heap_size_hint_in_bytes::Union{Integer,Nothing}=nothing,
    runtests::Bool=true,
    saved_state=nothing,
    return_state::Union{Bool,Nothing,Val}=nothing,
    loss_type::Type{L}=Nothing,
    verbosity::Union{Integer,Nothing}=nothing,
    progress::Union{Bool,Nothing}=nothing,
    X_units::Union{AbstractVector,Nothing}=nothing,
    y_units=nothing,
    v_dim_out::Val{DIM_OUT}=Val(nothing),
    # Deprecated:
    multithreaded=nothing,
    varMap=nothing,
) where {T<:DATA_TYPE,L,DIM_OUT}
    if multithreaded !== nothing
        error(
            "`multithreaded` is deprecated. Use the `parallelism` argument instead. " *
            "Choose one of :multithreaded, :multiprocessing, or :serial.",
        )
    end
    variable_names = deprecate_varmap(variable_names, varMap, :equation_search)

    if weights !== nothing
        @assert length(weights) == length(y)
        weights = reshape(weights, size(y))
    end

    datasets = construct_datasets(
        X,
        y,
        weights,
        variable_names,
        display_variable_names,
        y_variable_names,
        X_units,
        y_units,
        L,
    )

    return equation_search(
        datasets;
        niterations=niterations,
        options=options,
        parallelism=parallelism,
        numprocs=numprocs,
        procs=procs,
        addprocs_function=addprocs_function,
        heap_size_hint_in_bytes=heap_size_hint_in_bytes,
        runtests=runtests,
        saved_state=saved_state,
        return_state=return_state,
        verbosity=verbosity,
        progress=progress,
        v_dim_out=Val(DIM_OUT),
    )
end

function equation_search(
    X::AbstractMatrix{T1}, y::AbstractMatrix{T2}; kw...
) where {T1<:DATA_TYPE,T2<:DATA_TYPE}
end

function equation_search(
    X::AbstractMatrix{T1}, y::AbstractVector{T2}; kw...
) where {T1<:DATA_TYPE,T2<:DATA_TYPE}
    return equation_search(X, reshape(y, (1, size(y, 1))); kw..., v_dim_out=Val(1))
end

function equation_search(dataset::Dataset; kws...)
end

function equation_search(
    datasets::Vector{D};
    niterations::Int=10,
    options::Options=Options(),
    parallelism=:multithreading,
    numprocs::Union{Int,Nothing}=nothing,
    procs::Union{Vector{Int},Nothing}=nothing,
    addprocs_function::Union{Function,Nothing}=nothing,
    heap_size_hint_in_bytes::Union{Integer,Nothing}=nothing,
    runtests::Bool=true,
    saved_state=nothing,
    return_state::Union{Bool,Nothing,Val}=nothing,
    verbosity::Union{Int,Nothing}=nothing,
    progress::Union{Bool,Nothing}=nothing,
    v_dim_out::Val{DIM_OUT}=Val(nothing),
) where {DIM_OUT,T<:DATA_TYPE,L<:LOSS_TYPE,D<:Dataset{T,L}}
    concurrency = if parallelism in (:multithreading, "multithreading")
        :multithreading
    elseif parallelism in (:multiprocessing, "multiprocessing")
        :multiprocessing
    elseif parallelism in (:serial, "serial")
        :serial
    else
        error(
            "Invalid parallelism mode: $parallelism. " *
            "You must choose one of :multithreading, :multiprocessing, or :serial.",
        )
        :serial
    end
    not_distributed = concurrency in (:multithreading, :serial)
    not_distributed &&
        procs !== nothing &&
        error(
            "`procs` should not be set when using `parallelism=$(parallelism)`. Please use `:multiprocessing`.",
        )
    not_distributed &&
        numprocs !== nothing &&
        error(
            "`numprocs` should not be set when using `parallelism=$(parallelism)`. Please use `:multiprocessing`.",
        )

    _return_state = if return_state isa Val
        first(typeof(return_state).parameters)
    else
        if options.return_state === nothing
            return_state === nothing ? false : return_state
        else
            @assert(
                return_state === nothing,
                "You cannot set `return_state` in both the `Options` and in the passed arguments."
            )
            options.return_state
        end
    end

    dim_out = if DIM_OUT === nothing
        length(datasets) > 1 ? 2 : 1
    else
        DIM_OUT
    end
    _numprocs::Int = if numprocs === nothing
        if procs === nothing
            4
        else
            length(procs)
        end
    else
        if procs === nothing
            numprocs
        else
            @assert length(procs) == numprocs
            numprocs
        end
    end

    _verbosity = if verbosity === nothing && options.verbosity === nothing
        1
    elseif verbosity === nothing && options.verbosity !== nothing
        options.verbosity
    elseif verbosity !== nothing && options.verbosity === nothing
        verbosity
    else
        error(
            "You cannot set `verbosity` in both the search parameters `Options` and the call to `equation_search`.",
        )
        1
    end
    _progress::Bool = if progress === nothing && options.progress === nothing
        (_verbosity > 0) && length(datasets) == 1
    elseif progress === nothing && options.progress !== nothing
        options.progress
    elseif progress !== nothing && options.progress === nothing
        progress
    else
        error(
            "You cannot set `progress` in both the search parameters `Options` and the call to `equation_search`.",
        )
        false
    end

    _addprocs_function = addprocs_function === nothing ? addprocs : addprocs_function

    exeflags = if VERSION >= v"1.9" && concurrency == :multiprocessing
        heap_size_hint_in_megabytes = floor(
            Int, (
                if heap_size_hint_in_bytes === nothing
                    (Sys.free_memory() / _numprocs)
                else
                    heap_size_hint_in_bytes
                end
            ) / 1024^2
        )
        _verbosity > 0 &&
            heap_size_hint_in_bytes === nothing &&
            @info "Automatically setting `--heap-size-hint=$(heap_size_hint_in_megabytes)M` on each Julia process. You can configure this with the `heap_size_hint_in_bytes` parameter."

        `--heap-size=$(heap_size_hint_in_megabytes)M`
    else
        ``
    end

    # Underscores here mean that we have mutated the variable
    return _equation_search(
        datasets,
        RuntimeOptions{concurrency,dim_out,_return_state}(;
            niterations=niterations,
            total_cycles=options.populations * niterations,
            numprocs=_numprocs,
            init_procs=procs,
            addprocs_function=_addprocs_function,
            exeflags=exeflags,
            runtests=runtests,
            verbosity=_verbosity,
            progress=_progress,
        ),
        options,
        saved_state,
    )
end

@noinline function _equation_search(
    datasets::Vector{D}, ropt::RuntimeOptions, options::Options, saved_state
) where {D<:Dataset}
    _validate_options(datasets, ropt, options)
    state = _create_workers(datasets, ropt, options)
    _initialize_search!(state, datasets, ropt, options, saved_state)
    _warmup_search!(state, datasets, ropt, options)
    _main_search_loop!(state, datasets, ropt, options)
    _tear_down!(state, ropt, options)
    return _format_output(state, ropt)
end

function _validate_options(
    datasets::Vector{D}, ropt::RuntimeOptions, options::Options
) where {T,L,D<:Dataset{T,L}}
    example_dataset = first(datasets)
    nout = length(datasets)
    @assert nout >= 1
    @assert (nout == 1 || ropt.dim_out == 2)
    @assert options.populations >= 1
    if ropt.progress
        @assert(nout == 1, "You cannot display a progress bar for multi-output searches.")
        @assert(ropt.verbosity > 0, "You cannot display a progress bar with `verbosity=0`.")
    end
    if options.node_type <: GraphNode && ropt.verbosity > 0
        @warn "The `GraphNode` interface and mutation operators are experimental and will change in future versions."
    end
    if ropt.runtests
        test_option_configuration(ropt.parallelism, datasets, options, ropt.verbosity)
        test_dataset_configuration(example_dataset, options, ropt.verbosity)
    end
    for dataset in datasets
        update_baseline_loss!(dataset, options)
    end
    if options.define_helper_functions
        set_default_variable_names!(first(datasets).variable_names)
    end
    if options.seed !== nothing
        seed!(options.seed)
    end
    return nothing
end
function _create_workers(
    datasets::Vector{D}, ropt::RuntimeOptions, options::Options
) where {T,L,D<:Dataset{T,L}}
    stdin_reader = watch_stream(stdin)

    record = RecordType()
    @recorder record["options"] = "$(options)"

    nout = length(datasets)
    example_dataset = first(datasets)
    NT = with_type_parameters(options.node_type, T)
    PopType = Population{T,L,NT}
    HallOfFameType = HallOfFame{T,L,NT}
    WorkerOutputType = get_worker_output_type(
        Val(ropt.parallelism), PopType, HallOfFameType
    )
    ChannelType = ropt.parallelism == :multiprocessing ? RemoteChannel : Channel

    # Pointers to populations on each worker:
    worker_output = Vector{WorkerOutputType}[WorkerOutputType[] for j in 1:nout]
    # Initialize storage for workers
    tasks = [Task[] for j in 1:nout]
    # Set up a channel to send finished populations back to head node
    channels = [[ChannelType(1) for i in 1:(options.populations)] for j in 1:nout]
    (procs, we_created_procs) = if ropt.parallelism == :multiprocessing
        configure_workers(;
            procs=ropt.init_procs,
            ropt.numprocs,
            ropt.addprocs_function,
            options,
            project_path=splitdir(Pkg.project().path)[1],
            file=@__FILE__,
            ropt.exeflags,
            ropt.verbosity,
            example_dataset,
            ropt.runtests,
        )
    else
        Int[], false
    end
    # Get the next worker process to give a job:
    worker_assignment = WorkerAssignments()
    # Randomly order which order to check populations:
    # This is done so that we do work on all nout equally.
    task_order = [(j, i) for j in 1:nout for i in 1:(options.populations)]
    shuffle!(task_order)

    # Persistent storage of last-saved population for final return:
    last_pops = init_dummy_pops(options.populations, datasets, options)
    # Best 10 members from each population for migration:
    best_sub_pops = init_dummy_pops(options.populations, datasets, options)
    # TODO: Should really be one per population too.
    all_running_search_statistics = [
        RunningSearchStatistics(; options=options) for j in 1:nout
    ]
    # Records the number of evaluations:
    # Real numbers indicate use of batching.
    num_evals = [[0.0 for i in 1:(options.populations)] for j in 1:nout]

    halls_of_fame = Vector{HallOfFameType}(undef, nout)

    cycles_remaining = [ropt.total_cycles for j in 1:nout]
    cur_maxsizes = [
        get_cur_maxsize(; options, ropt.total_cycles, cycles_remaining=cycles_remaining[j])
        for j in 1:nout
    ]

    return SearchState{
        T,L,with_type_parameters(options.node_type, T),WorkerOutputType,ChannelType
    }(;
        procs=procs,
        we_created_procs=we_created_procs,
        worker_output=worker_output,
        tasks=tasks,
        channels=channels,
        worker_assignment=worker_assignment,
        task_order=task_order,
        halls_of_fame=halls_of_fame,
        last_pops=last_pops,
        best_sub_pops=best_sub_pops,
        all_running_search_statistics=all_running_search_statistics,
        num_evals=num_evals,
        cycles_remaining=cycles_remaining,
        cur_maxsizes=cur_maxsizes,
        stdin_reader=stdin_reader,
        record=Ref(record),
    )
end
function _initialize_search!(
    state::SearchState{T,L,N}, datasets, ropt::RuntimeOptions, options::Options, saved_state
) where {T,L,N}
    nout = length(datasets)

    init_hall_of_fame = load_saved_hall_of_fame(saved_state)
    if init_hall_of_fame === nothing
        for j in 1:nout
            state.halls_of_fame[j] = HallOfFame(options, T, L)
        end
    else
        # Recompute losses for the hall of fame, in
        # case the dataset changed:
        for j in eachindex(init_hall_of_fame, datasets, state.halls_of_fame)
            hof = init_hall_of_fame[j]
            for member in hof.members[hof.exists]
                score, result_loss = score_func(datasets[j], member, options)
                member.score = score
                member.loss = result_loss
            end
            state.halls_of_fame[j] = hof
        end
    end

    for j in 1:nout, i in 1:(options.populations)
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )
        saved_pop = load_saved_population(saved_state; out=j, pop=i)
        new_pop =
            if saved_pop !== nothing && length(saved_pop.members) == options.population_size
                saved_pop::Population{T,L,N}
                ## Update losses:
                for member in saved_pop.members
                    score, result_loss = score_func(datasets[j], member, options)
                    member.score = score
                    member.loss = result_loss
                end
                copy_pop = copy(saved_pop)
                @sr_spawner(
                    begin
                        (copy_pop, HallOfFame(options, T, L), RecordType(), 0.0)
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
            else
                if saved_pop !== nothing && ropt.verbosity > 0
                    @warn "Recreating population (output=$(j), population=$(i)), as the saved one doesn't have the correct number of members."
                end
                @sr_spawner(
                    begin
                        (
                            Population(
                                datasets[j];
                                population_size=options.population_size,
                                nlength=3,
                                options=options,
                                nfeatures=datasets[j].nfeatures,
                            ),
                            HallOfFame(options, T, L),
                            RecordType(),
                            Float64(options.population_size),
                        )
                    end,
                    parallelism = ropt.parallelism,
                    worker_idx = worker_idx
                )
                # This involves population_size evaluations, on the full dataset:
            end
        push!(state.worker_output[j], new_pop)
    end
    return nothing
end
function _warmup_search!(
    state::SearchState{T,L,N}, datasets, ropt::RuntimeOptions, options::Options
) where {T,L,N}
    nout = length(datasets)
    for j in 1:nout, i in 1:(options.populations)
        dataset = datasets[j]
        running_search_statistics = state.all_running_search_statistics[j]
        cur_maxsize = state.cur_maxsizes[j]
        @recorder state.record[]["out$(j)_pop$(i)"] = RecordType()
        worker_idx = assign_next_worker!(
            state.worker_assignment; out=j, pop=i, parallelism=ropt.parallelism, state.procs
        )

        # TODO - why is this needed??
        # Multi-threaded doesn't like to fetch within a new task:
        c_rss = deepcopy(running_search_statistics)
        last_pop = state.worker_output[j][i]
        updated_pop = @sr_spawner(
            begin
                in_pop = first(
                    extract_from_worker(last_pop, Population{T,L,N}, HallOfFame{T,L,N})
                )
                _dispatch_s_r_cycle(
                    in_pop,
                    dataset,
                    options;
                    pop=i,
                    out=j,
                    iteration=0,
                    ropt.verbosity,
                    cur_maxsize,
                    running_search_statistics=c_rss,
                )::DefaultWorkerOutputType{Population{T,L,N},HallOfFame{T,L,N}}
            end,
            parallelism = ropt.parallelism,
            worker_idx = worker_idx
        )
        state.worker_output[j][i] = updated_pop
    end
    return nothing
end

function _main_search_loop!(
    state::SearchState{T,L,N}, datasets, ropt::RuntimeOptions, options::Options
) where {T,L,N}
end
function _tear_down!(state::SearchState, ropt::RuntimeOptions, options::Options)
end
function _format_output(state::SearchState, ropt::RuntimeOptions)
end

function _dispatch_s_r_cycle(
    in_pop::Population{T,L,N},
    dataset::Dataset,
    options::Options;
    pop::Int,
    out::Int,
    iteration::Int,
    verbosity,
    cur_maxsize::Int,
    running_search_statistics,
) where {T,L,N}
    record = RecordType()
    # @recorder record["out$(out)_pop$(pop)"] = RecordType(
    #     "iteration$(iteration)" => record_population(in_pop, options)
    # )
    num_evals = 0.0
    # normalize_frequencies!(running_search_statistics)
    out_pop, best_seen, evals_from_cycle = s_r_cycle(
        dataset,
        in_pop,
        options.ncycles_per_iteration,
        cur_maxsize,
        running_search_statistics;
        verbosity=verbosity,
        options=options,
        record=record,
    )
    # num_evals += evals_from_cycle
    out_pop, evals_from_optimize = optimize_and_simplify_population(
        dataset, out_pop, options, cur_maxsize, record
    )
    # num_evals += evals_from_optimize
    # if options.batching
    #     for i_member in 1:(options.maxsize + MAX_DEGREE)
    #         score, result_loss = score_func(dataset, best_seen.members[i_member], options)
    #         best_seen.members[i_member].score = score
    #         best_seen.members[i_member].loss = result_loss
    #         num_evals += 1
    #     end
    # end
    return (out_pop, best_seen, record, num_evals)
end


using PrecompileTools: @compile_workload, @setup_workload
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
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
end

end #module SR
