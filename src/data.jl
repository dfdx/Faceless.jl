

Base.convert{T}(::Type{Matrix{Float64}}, img::Image{Gray{T}}) =
    convert(Array{Float64, 2}, convert(Array, img))


function walkdir(dir)
    paths = [joinpath(dir, filename) for filename in readdir(dir)]
    result = Array(String, 0)
    for path in paths
        if isdir(path)
            append!(result, walkdir(path))
        else
            push!(result, path)
        end
    end
    return result
end


imgdir = expanduser("~/data/CK/images")
shapedir = expanduser("~/data/CK/landmarks")


function read_images_ck(imgdir::String; start=1, count=-1)
    filenames = sort(readdir(imgdir))
    paths = [joinpath(imgdir, filename) for filename in filenames]
    num = count != -1 ? count : length(paths)  # if count == -1, read all
    num = min(num, length(paths) - start + 1)  # don't cross the bounds
    imgs = Array(Matrix{Float64}, num)
    for i=1:num
        img = imread(paths[start + i - 1])
        if colorspace(img) != "Gray"
            img = convert(Array{Gray}, img)
        end
        imgs[i] = convert(Matrix{Float64}, img)
        if i % 100 == 0
            info("$i images read")
        end
    end
    return imgs
end


function read_shape_ck(path::String)
    open(path) do file
        readdlm(file)
    end
end


function read_shapes_ck(shapedir::String; start=1, count=-1)
    filenames = sort(readdir(shapedir))
    paths = [joinpath(shapedir, filename) for filename in filenames]
    num = count != -1 ? count : length(paths)  # if count == -1, read all
    num = min(num, length(paths) - start + 1)  # don't cross the bounds
    shapes = Array(Matrix{Float64}, num)
    for i=1:num
        shape_xy = read_shape_ck(paths[start + i - 1])
        shapes[i] = [shape_xy[:, 2] shape_xy[:, 1]]
    end
    return shapes
end
