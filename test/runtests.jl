using MLFlowLogger
using Test

function cleanup_experiment(mlflogger)
    experiment_id = mlflogger.run.info.experiment_id
    deleteexperiment(mlflogger.mlf, experiment_id)
end

"""
The tests require that ENV["MLFLOW_URI"] is set to a valid uri
and that a mlflow server is running at that uri.
"""

include("constructors.jl")
include("logging.jl")
