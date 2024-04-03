using Base.CoreLogging
using FileIO

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

@testset "Log Parameter" begin
    mlflogger = MLFLogger(tracking_uri=ENV["MLFLOW_URI"])
    key = "very specific parameter key"
    value = "parameter value"  # is a String in MLFlowClient.MLFlowRunDataParam
    MLFlowLogger.log_param(mlflogger, key, value)

    experiment_id = mlflogger.run.info.experiment_id
    retrieved_runs = searchruns(mlflogger.mlf, experiment_id)

    @test retrieved_runs[1].data.params[key].value == value

    cleanup_experiment(mlflogger)
end

@testset "Log Artifact" begin
    mlflogger = MLFLogger(tracking_uri=ENV["MLFLOW_URI"])
    filepath = "artifact.txt"
    write(filepath, "hi I am an artifact file")

    artifact = MLFlowLogger.log_artifact(mlflogger, filepath)
    @test isfile(artifact)

    cleanup_experiment(mlflogger)
    rm(filepath)
end

@testset "Log Image" begin
    mlflogger = MLFLogger(tracking_uri=ENV["MLFLOW_URI"])
    arr = rand(4,3)
    filepath = "image.png"
    save(filepath, arr)

    artifact = MLFlowLogger.log_image(mlflogger, filepath)
    @test isfile(artifact)

    #cleanup_experiment(mlflogger)
    rm(filepath)
end

@testset "Core Logging" begin
    mlflogger = MLFLogger(
        tracking_uri=ENV["MLFLOW_URI"],
        min_level=CoreLogging.Debug
        )
    @test CoreLogging.min_enabled_level(mlflogger) == CoreLogging.Debug

    struct sample_struct
        integer::Int
        float::Float64
    end

    with_logger(mlflogger) do
        @test mlflogger.global_step == 0
        @info "group 1" accuracy=0.8 loss=0.4 complex=1+2im
        @test mlflogger.global_step == 1
        @info "group 1" accuracy=0.9 loss=0.2 complex=1+1im
        @test mlflogger.global_step == 2

        sample = sample_struct(4, 3.14159265359)
        @info "log struct" sample=sample  log_step_increment=0
        @test mlflogger.global_step == 2
    end

    experiment_id = mlflogger.run.info.experiment_id
    retrieved_runs = searchruns(mlflogger.mlf, experiment_id)

    @test retrieved_runs[1].data.metrics["group 1/accuracy"].value == 0.9
    @test retrieved_runs[1].data.metrics["group 1/accuracy"].step == 2
    @test retrieved_runs[1].data.metrics["group 1/complex/im"].value == 1
    @test retrieved_runs[1].data.metrics["log struct/sample/integer"].value == 4
    @test retrieved_runs[1].data.metrics["log struct/sample/float"].value == 3.14159265359
    @test retrieved_runs[1].data.metrics["log struct/sample/integer"].step == 2

    cleanup_experiment(mlflogger)
end
