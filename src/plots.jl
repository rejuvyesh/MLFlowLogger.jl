using .Plots

function preprocess(name, plot::Plots.Plot, data)
#=     pb = PipeBuffer()
    show(pb, MIME("image/png"), plot)
    arr = FileIO.load(pb)
 =#    push!(data, name=>plot)
    return data
end

function preprocess(name, plots::AbstractArray{<:Plots.Plot}, data)
    for (i, plot)=enumerate(plots)
        preprocess(name*"/$i", plot, data)
    end
    return data
end


function log_plot(lg::MLFLogger, key::AbstractString, obj::Plots.Plot; step=nothing)
    mktempdir(;prefix="jlmlf_") do dirname
        fname = joinpath(dirname, key)
        mkpath(fname)
        fname = joinpath(fname, "plot.png")
        savefig(obj, fname)
        log_image(lg, key, fname; step=step)
    end

end

process(lg::MLFLogger, name::AbstractString, obj::Plots.Plot, step::Int) = log_plot(lg, name, obj; step=step)