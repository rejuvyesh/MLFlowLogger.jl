module MLFlowLogger

using PyCall
using UUIDs
using Requires
using FileIO
using ColorTypes

export MLFLogger

const mlflow = PyNULL()

function __init__()
    copy!(mlflow, pyimport("mlflow"))
    @require ImageIO="82e4d734-157c-48bb-816b-45c225c6df19" begin    
        include("image.jl")
        @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
            include("plots.jl")
        end    
    end
end

using Base.CoreLogging: CoreLogging, AbstractLogger, LogLevel, Info, handle_message, shouldlog, min_enabled_level, catch_exceptions

mutable struct MLFLogger <: AbstractLogger
    client::PyObject
    run::PyObject
    step_increment::Int
    global_step::Int
    min_level::LogLevel
end

function MLFLogger(; min_level=Info, step_increment=1, start_step=0, experiment_name=nothing, tracking_uri=nothing, kwargs...)
    if isnothing(tracking_uri)
        client = mlflow.tracking.MlflowClient()
    else
        client = mlflow.tracking.MlflowClient(tracking_uri)
    end

    expid = nothing
    if haskey(ENV, "MLFLOW_EXPERIMENT_ID")
        expid = ENV["MLFLOW_EXPERIMENT_ID"]
        exp = client.get_experiment(expid)
        experiment_name = exp.name
    end
    
    if isnothing(experiment_name)
        experiment_name = string(UUIDs.uuid4())
    end
    if isnothing(expid)
        expid = client.create_experiment(experiment_name)
    end

    if haskey(ENV, "MLFLOW_RUN_ID")
        runid = ENV["MLFLOW_RUN_ID"]
        run = client.get_run(runid)
    else
        run = client.create_run(expid)    
    end
    
    MLFLogger(client, run, step_increment, start_step, min_level)
end

increment_step!(lg::MLFLogger, Δ_Step) = lg.global_step += Δ_Step

add_tag!(lg::MLFLogger, tag::String) = lg.client.set_experiment_tag(lg.run.info.experiment_id, "tag", tag)
add_tag!(lg::MLFLogger, key::String, value::String) = lg.client.set_experiment_tag(lg.run.info.experiment_id, key, value)


"""
    function log_metric(lg::CLogger, name::AbstractString, value::Real; step::Int=nothing, epoch::Int=nothing)
        
Logs general scalar metrics.        
"""
function log_metric(lg::MLFLogger, key::AbstractString, value; timestamp=nothing, step=nothing)
    lg.client.log_metric(lg.run.info.run_id, key, value, timestamp=timestamp, step=step)
end

function log_param(lg::MLFLogger, key::AbstractString, value)
    lg.client.log_param(lg.run.info.run_id, key, value)
end

"""
    function log_artifact(lg::MLFLogger, local_path, artifact_path)

Log a local file or directory as an artifact of the currently active run.

- `local_path`: Path to the file to write
- `artifact_path`: If provided, the directory to write to.
"""
function log_artifact(lg::MLFLogger, local_path, artifact_path=nothing)
    lg.client.log_artifact(lg.run.info.run_id, local_path, artifact_path)
end

"""
    function log_image(lg::MLFLogger, key::AbstractString, obj::AbstractString; step=nothing)

Log a local image file as an artifact of the currently active run. If current `step` provided, adds
it to the artifact directory path.
"""
function log_image(lg::MLFLogger, key::AbstractString, obj::AbstractString; step=nothing)
    if !ispath(obj)
        @warn "$obj not a path to an image"
        return        
    end
    if !isnothing(step)
        key = key * "/$step"
    end
    log_artifact(lg, obj, key)
end

const log_parameter = log_param

CoreLogging.catch_exceptions(lg::MLFLogger) = false

CoreLogging.min_enabled_level(lg::MLFLogger) = lg.min_level

CoreLogging.shouldlog(lg::MLFLogger, level, _module, group, id) = true

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

"""
    logable_propertynames(val::Any)
Returns a tuple with the name of the fields of the structure `val` that
should be logged to Comet.ml. This function should be overridden when
you want Comet.ml to ignore some fields in a structure when logging
it. The default behaviour is to return the  same result as `propertynames`.
See also: [`Base.propertynames`](@ref)
"""
logable_propertynames(val::Any) = propertynames(val)

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
process(lg::MLFLogger, name::AbstractString, obj::Real, step::Int) = log_metric(lg, name, obj; step=step)

function CoreLogging.handle_message(lg::MLFLogger, level, message, _module, group, id, file, line; kwargs...)
    i_step = lg.step_increment # :log_step_increment default value

    if !isempty(kwargs)
        data = Vector{Pair{String,Any}}()
        # ∀ (k-v) pairs, decompose values into objects that can be serialized
        for (key,val) in pairs(kwargs)
            # special key describing step increment
            if key == :log_step_increment
                i_step = val
                continue
            end
            preprocess(message*"/$key", val, data)
        end
        iter = increment_step!(lg, i_step)
        for (name, val) in data
            process(lg, name, val, iter)
        end        
    end    
end

Base.show(io::IO, lg::MLFLogger) = begin
	str  = "MLFLogger(\"run=$(lg.run.info.run_id)\", min_level=$(lg.min_level), "*
		   "current_step=$(lg.global_step))"
    Base.print(io, str)
end

Base.close(lg::MLFLogger) = lg.client.set_terminated(lg.run.info.run_id)

end
