using .ImageIO

function log_image(lg::MLFLogger, key::AbstractString, obj::AbstractArray{<:Colorant}; step=nothing)
    mktempdir(;prefix="jlmlf_") do dirname
        fname = joinpath(dirname, key)
        fname = joinpath(fname, "image.png")
        FileIO.save(fname, obj)
        log_image(lg, key, fname; step=step)
    end
end

process(lg::MLFLogger, name::AbstractString, obj::AbstractArray{<:Colorant}, step::Int) = log_image(lg, name, obj; step=step)