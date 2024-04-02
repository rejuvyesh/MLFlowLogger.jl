@testset "Increment Step" begin
    mlflogger = MLFLogger(
        tracking_uri=ENV["MLFLOW_URI"],
        start_step=0,
        step_increment=1
        )
    @test mlflogger.global_step == 0

    MLFlowLogger.increment_step!(mlflogger, mlflogger.step_increment)
    @test mlflogger.global_step == 1

    cleanup_experiment(mlflogger)
end

@testset "Log Metric" begin
    mlflogger = MLFLogger(tracking_uri=ENV["MLFLOW_URI"])
    key = "very specific key"
    value = 42
    MLFlowLogger.log_metric(mlflogger, key, value)

    experiment_id = mlflogger.run.info.experiment_id
    retrieved_runs = searchruns(mlflogger.mlf, experiment_id)

    @test retrieved_runs[1].data.metrics[key].key == key
    @test retrieved_runs[1].data.metrics[key].value == value

    cleanup_experiment(mlflogger)
end