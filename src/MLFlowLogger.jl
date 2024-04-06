module MLFlowLogger

using MLFlowClient
using UUIDs
using Base.CoreLogging: CoreLogging, AbstractLogger, LogLevel, handle_message, shouldlog, min_enabled_level, catch_exceptions

export MLFLogger

mutable struct MLFLogger <: AbstractLogger
    mlf::MLFlow
    run::MLFlowRun
    global_step::Int
    step_increment::Int
    min_level::LogLevel
end

function MLFLogger(;
    tracking_uri=nothing, 
    experiment_name=nothing,
    run_id=nothing,
    start_step=0,
    step_increment=1,
    min_level=CoreLogging.Info,
    kwargs...)

    if isnothing(tracking_uri)
        mlf = MLFlow()
    else
        mlf = MLFlow(tracking_uri, kwargs...)
    end

    if isnothing(experiment_name)
        experiment_name = string(UUIDs.uuid4())
    end
    experiment = getorcreateexperiment(mlf, experiment_name, kwargs...)

    if isnothing(run_id)
        run = createrun(mlf, experiment.experiment_id, kwargs...)
    else
        run = getrun(mlf, run_id, kwargs...)
    end

    MLFLogger(mlf, run, start_step, step_increment, min_level)
end

increment_step!(logger::MLFLogger, Δ_Step) = logger.global_step += Δ_Step

"""
    function log_metric(logger::MLFLogger, key::AbstractString, value::Real; timestamp=missing, step=missing)
        
Logs general scalar metrics.        
"""
function log_metric(logger::MLFLogger, key::AbstractString, value::Real; timestamp=missing, step=missing)
    logmetric(logger.mlf, logger.run, key, value, timestamp=timestamp, step=step)
end

function log_param(logger::MLFLogger, key::AbstractString, value)
    logparam(logger.mlf, logger.run, key, value)
end

"""
    function log_artifact(logger::MLFLogger, filepath)

Log a local file as an artifact of the currently active run.

- `filepath`: Path to the file
"""
function log_artifact(logger::MLFLogger, filepath)
    logartifact(logger.mlf, logger.run, filepath)
end

"""
    function log_image(logger::MLFLogger, obj::AbstractString)

Log a local image file as an artifact of the currently active run.
"""
function log_image(logger::MLFLogger, obj::AbstractString)
    if !ispath(obj)
        @warn "$obj not a path to an image"
        return
    end
    log_artifact(logger, obj)
end

CoreLogging.catch_exceptions(logger::MLFLogger) = false

CoreLogging.min_enabled_level(logger::MLFLogger) = logger.min_level

CoreLogging.shouldlog(logger::MLFLogger, level, _module, group, id) = true

"""
    logable_propertynames(val::Any)
Returns a tuple with the name of the fields of the structure `val` that
should be logged to Comet.ml. This function should be overridden when
you want Comet.ml to ignore some fields in a structure when logging
it. The default behaviour is to return the  same result as `propertynames`.
See also: [`Base.propertynames`](@extref)
"""
logable_propertynames(val::Any) = propertynames(val)

function preprocess(name, val::T, data) where {T}
    if isstructtype(T)
        fn = logable_propertynames(val)
        for f=fn
            prop = getfield(val, f)
            preprocess(name*"/$f", prop, data)
        end
    else
        # If we do not know how to serialize a type, then
        # it will be simply logged as text        
        push!(data, name=>val)
    end
    data
end

## Default unpacking of key-value dictionaries
function preprocess(name, dict::AbstractDict, data)
    for (key, val) in dict
        # convert any key into a string, via interpolating it
        preprocess("$name/$key", val, data)
    end
    return data
end

# Split complex numbers into real/complex pairs
preprocess(name, val::Complex, data) = push!(data, name*"/re"=>real(val), name*"/im"=>imag(val))

# Handle standard float metrics
process(logger::MLFLogger, name::AbstractString, obj::Real, step::Int) = log_metric(logger, name, obj; step=step)

function CoreLogging.handle_message(logger::MLFLogger, level, message, _module, group, id, file, line; kwargs...)
    i_step = logger.step_increment

    if !isempty(kwargs)
        data = Vector{Pair{String,Any}}()
        # ∀ (k-v) pairs, decompose values into objects that can be serialized
        for (key,val) in pairs(kwargs)
            # special key describing step increment
            if key == :log_step_increment
                i_step = val
                continue
            end
            preprocess(strip(message*"/$key", ['(', ')', '{', '}', '!']), val, data)
        end
        iter = increment_step!(logger, i_step)
        for (name, val) in data
            process(logger, name, val, iter)
        end        
    end    
end

Base.show(io::IO, logger::MLFLogger) = begin
	str  = "MLFLogger("*
        "tracking_uri=$(logger.mlf.baseuri), "*
        "experiment=$(logger.run.info.experiment_id), "*
        "run=$(logger.run.info.run_id), "*
        "status=$(logger.run.info.status.status), "*
        "min_level=$(logger.min_level)"*
        ")"
    Base.print(io, str)
end

end
