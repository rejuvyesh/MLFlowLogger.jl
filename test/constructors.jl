using MLFlowClient

@testset "Default Constructor" begin
    mlflogger = MLFLogger(tracking_uri=ENV["MLFLOW_URI"])
    @test isa(mlflogger, MLFLogger)

    cleanup_experiment(mlflogger)
end

@testset "Constructor Arguments" begin
    mlflogger = MLFLogger(
        tracking_uri=ENV["MLFLOW_URI"],
        experiment_name="very specific experiment name"
        )
    @test isa(mlflogger, MLFLogger)
    @test mlflogger.mlf.baseuri == ENV["MLFLOW_URI"]
    experiment_id = mlflogger.run.info.experiment_id
    @test MLFlowClient.getexperiment(
        mlflogger.mlf, 
        experiment_id
        ).name == "very specific experiment name"

    # the run_id argument is tested separately because it requires an existing run_id
    run_id = mlflogger.run.info.run_id
    mlflogger_same_run = MLFLogger(
        tracking_uri=ENV["MLFLOW_URI"],
        experiment_name="very specific experiment name",
        run_id=run_id
        )
    @test isa(mlflogger_same_run, MLFLogger)
    @test mlflogger_same_run.run.info.run_id == run_id

    cleanup_experiment(mlflogger)
end