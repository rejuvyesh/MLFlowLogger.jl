using PyCall

try
    pyimport("mlflow")
catch e
    try
        run(`$(PyCall.pyprogramname) -m pip install mlflow`)
    catch ee
        if !(typeof(ee) <: PyCall.PyError)
            rethrow(ee)
        end
        @warn("""
    Python dependencies not installed.
    Either
    - Rebuild `PyCall` to use Conda by running the following in Julia REPL
        - `ENV[PYTHON]=""; using Pkg; Pkg.build("PyCall"); Pkg.build("MLFlowLogger")
    - Or install the dependencies by running `pip`
        - `pip install mlflow`
        """
              )              
    end
end