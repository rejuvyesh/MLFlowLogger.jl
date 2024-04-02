module MLFlowLogger

using MLFlowClient
using UUIDs

export MLFLogger

mutable struct MLFLogger
    mlf::MLFlow
    run::MLFlowRun
    global_step::Int
    step_increment::Int
end

function MLFLogger(;
    tracking_uri=nothing, 
    experiment_name=nothing,
    run_id=nothing,
    start_step=0,
    step_increment=1,
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

    MLFLogger(mlf, run, start_step, step_increment)
end

increment_step!(logger::MLFLogger, Δ_Step) = logger.global_step += Δ_Step

"""
    function log_metric(logger::MLFLogger, key::AbstractString, value::Real; timestamp=missing, step=missing)
        
Logs general scalar metrics.        
"""
function log_metric(logger::MLFLogger, key::AbstractString, value::Real; timestamp=missing, step=missing)
    logmetric(logger.mlf, logger.run, key, value, timestamp=timestamp, step=step)
end

Base.show(io::IO, logger::MLFLogger) = begin
	str  = "MLFLogger("*
        "tracking_uri=$(logger.mlf.baseuri), "*
        "experiment=$(logger.run.info.experiment_id), "*
        "run=$(logger.run.info.run_id), "*
        "status=$(logger.run.info.status.status)"
    Base.print(io, str)
end

end
