module AdaptiveParsimonyModule

using ..CoreModule: Options, MAX_DEGREE

mutable struct RunningSearchStatistics
    window_size::Int
    frequencies::Vector{Float64}
    normalized_frequencies::Vector{Float64}  # Stores `frequencies`, but normalized (updated once in a while)
end

function RunningSearchStatistics(; options::Options, window_size::Int=100000)
    maxsize = options.maxsize
    actualMaxsize = maxsize + MAX_DEGREE
    init_frequencies = ones(Float64, actualMaxsize)

    return RunningSearchStatistics(
        window_size, init_frequencies, copy(init_frequencies) / sum(init_frequencies)
    )
end

@inline function update_frequencies!(
    running_search_statistics::RunningSearchStatistics; size=nothing
)
    if 0 < size <= length(running_search_statistics.frequencies)
        running_search_statistics.frequencies[size] += 1
    end
    return nothing
end

function move_window!(running_search_statistics::RunningSearchStatistics)
    smallest_frequency_allowed = 1
    max_loops = 1000

    frequencies = running_search_statistics.frequencies
    window_size = running_search_statistics.window_size

    cur_size_frequency_complexities = sum(frequencies)
    if cur_size_frequency_complexities > window_size
        difference_in_size = cur_size_frequency_complexities - window_size
        num_loops = 0
        while difference_in_size > 0
            indices_to_subtract = findall(frequencies .> smallest_frequency_allowed)
            num_remaining = size(indices_to_subtract, 1)
            amount_to_subtract = min(
                difference_in_size / num_remaining,
                min(frequencies[indices_to_subtract]...) - smallest_frequency_allowed,
            )
            frequencies[indices_to_subtract] .-= amount_to_subtract
            total_amount_to_subtract = amount_to_subtract * num_remaining
            difference_in_size -= total_amount_to_subtract
            num_loops += 1
            if num_loops > max_loops || total_amount_to_subtract < 1e-6
                break
            end
        end
    end
    return nothing
end

function normalize_frequencies!(running_search_statistics::RunningSearchStatistics)
    running_search_statistics.normalized_frequencies .=
        running_search_statistics.frequencies ./ sum(running_search_statistics.frequencies)
    return nothing
end

end
