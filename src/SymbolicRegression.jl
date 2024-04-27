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

const PACKAGE_VERSION = VersionNumber(0, 0, 0)


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

    options::Options=Options(),
    parallelism=:serial,

    loss_type::Type{L}=Nothing,
) where {T<:DATA_TYPE,L,DIM_OUT}
    datasets = construct_datasets(
        X,
        y,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        L,
    )

    return equation_search(
        datasets;
        options=options,
        parallelism=parallelism,
    )
end

function equation_search(
    datasets::Vector{D};

    options::Options=Options(),
    parallelism=:multithreading,

    return_state::Union{Bool,Nothing,Val}=nothing,
    v_dim_out::Val{DIM_OUT}=Val(nothing),
) where {DIM_OUT,T<:DATA_TYPE,L<:LOSS_TYPE,D<:Dataset{T,L}}
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

    return _equation_search(
        datasets,
        RuntimeOptions{:serial, dim_out, _return_state}(;
            niterations=10,
            total_cycles=options.populations * 10,
            numprocs=4,
            init_procs=nothing,
            addprocs_function=addprocs,
            exeflags=``,
            runtests=true,
            verbosity=false,
            progress=false,
        ),
        options,
        nothing,
    )
end

@noinline function _equation_search(
    datasets::Vector{D}, ropt::RuntimeOptions, options::Options, saved_state
) where {D<:Dataset}
    state = _create_workers(datasets, ropt, options)
    _initialize_search!(state, datasets, ropt, options, saved_state)

    _warmup_search!(state, datasets, ropt, options)
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

    for j in 1:nout
        state.halls_of_fame[j] = HallOfFame(options, T, L)
    end

    for j in 1:nout, i in 1:(options.populations)
        new_pop =
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

        c_rss = deepcopy(running_search_statistics)
        last_pop = state.worker_output[j][i]
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
        )
    end
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

    num_evals = 0.0

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

    out_pop, evals_from_optimize = optimize_and_simplify_population(
        dataset, out_pop, options, cur_maxsize, record
    )

    return (out_pop, best_seen, record, num_evals)
end


using PrecompileTools: @compile_workload, @setup_workload
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        @setup_workload begin
            X = zeros(2, 1);
            Y = zeros(1, 1);
    
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
