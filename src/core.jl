
using PiecewiseAffineTransforms
using MultivariateStats
using ActiveAppearanceModels
using Images
using ImageView

include("data.jl")


# 4. apply PCA
# 5. apply RBM


function to_dataset(aam::AAModel, imgs::Vector{Matrix{Float64}},
                    shapes::Vector{Shape})
    warped0 = pa_warp(aam.wparams, img, shape)
    v0 = to_vector(warped0, aam.wparams.warp_map)
    dataset = zeros(Float64, length(v0), length(imgs))
    for i=1:length(imgs)
        warped = pa_warp(aam.wparams, imgs[i], shapes[i])
        dataset[:, i] = to_vector(warped, aam.wparams.warp_map)
    end
    dataset
end


function to_vector{T}(img::Matrix{T}, mask::Matrix{Int})
    vec = Array(T, 0)
    for j=1:size(img, 2)
        for i=1:size(img, 1)
            if mask[i, j] > 0
                push!(vec, img[i, j])
            end
        end
    end
    vec
end

function to_image{T}(vec::Vector{T}, mask::Matrix{Int})
    img = zeros(T, size(mask))
    k = 1
    for j=1:size(img, 2)
        for i=1:size(img, 1)
            if mask[i, j] > 0
                img[i, j] = vec[k]
                k += 1
            end
        end
    end
    img
end



function main()
    imgs = read_images_ck(imgdir, count=1000)
    shapes = read_shapes_ck(shapedir, count=1000)
    @time aam = train(AAModel(), imgs, shapes)

    dataset = to_dataset(aam, imgs, shapes)
    

    ## img = imgs[160]
    ## shape = shapes[160]
    ## warped = pa_warp(aam.wparams, img, shape)
    ## v = to_vector(warped, aam.wparams.warp_map)
    ## warped2 = to_image(v, aam.wparams.warp_map)
end
