
using PiecewiseAffineTransforms
using MultivariateStats
using ActiveAppearanceModels
using Images
using ImageView
using Color

include("data.jl")
include("view.jl")

# 4. apply PCA
# 5. apply RBM


function to_dataset(aam::AAModel, imgs::Vector{Matrix{Float64}},
                    shapes::Vector{Shape})
    warped1 = pa_warp(aam.wparams, imgs[1], shapes[1])
    v1 = to_vector(warped1, aam.wparams.warp_map)
    dataset = zeros(Float64, length(v1), length(imgs))
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


function normalize(img)
    mn, mx = minimum(img), maximum(img)
    nimg = (img .- mn) ./ (mx - mn)
    nimg
end


nview(img) = view(normalize(img))


function save_images{T,N}(imgs::Vector{Array{T,N}}, path::String; prefix="img")
    for i=1:length(imgs)
        imwrite(imgs[i], joinpath(path, @sprintf("%s_%03d.png", prefix, i)))
    end
end


function main()
    imgs = read_images_ck(DATA_DIR_CK, resizeratio=0.5)
    shapes = read_shapes_ck(DATA_DIR_CK, resizeratio=0.5)
    @time aam = train(AAModel(), imgs, shapes)

    dataset = to_dataset(aam, imgs, shapes)
    P = projection(fit(PCA, dataset))
    nview(to_image(P[:, 1], aam.wparams.warp_map))
    pca_imgs = Matrix{Float64}[normalize(to_image(P[:, i], aam.wparams.warp_map))
                               for i=1:size(P, 2)]
    save_images(pca_imgs,
                expanduser("~/Dropbox/PhD/MyPapers/facial_expr_repr/images/pca"))
end

