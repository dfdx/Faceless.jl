
function to_dataset{N}(wparams::PAWarpParams, imgs::Vector{Array{Float64,N}},
                    shapes::Vector{Shape})
    warped1 = pa_warp(wparams, imgs[1], shapes[1])
    v1 = to_vector(warped1, wparams.warp_map)
    dataset = zeros(Float64, length(v1), length(imgs))
    for i=1:length(imgs)
        warped = pa_warp(wparams, imgs[i], shapes[i])
        dataset[:, i] = to_vector(warped, wparams.warp_map)
    end
    dataset
end

## function to_dataset(wparams::PAWarpParams, imgs::Vector{Array{Float64,3}},
##                     shapes::Vector{Shape})
##     warped1 = pa_warp(wparams, imgs[1], shapes[1])
##     v1 = to_vector(warped1, wparams.warp_map)
##     dataset = zeros(Float64, length(v1), length(imgs))
##     for i=1:length(imgs)
##         warped = pa_warp(wparams, imgs[i], shapes[i])
##         dataset[:, i] = to_vector(warped, wparams.warp_map)
##     end
##     dataset
## end

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

function to_vector{T}(img::Array{T,3}, mask::Matrix{Int})
    vec = Array(T, 0)
    for k=1:size(img, 3)
        for j=1:size(img, 2)
            for i=1:size(img, 1)
                if mask[i, j] > 0
                    push!(vec, img[i, j])
                end
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


function load_image_dataset(name::Symbol)
    aam = load_aam()
    imgs = load_images(name, datadir=DATA_DIR_CK,
                       resizeratio=RESIZERATIO)
    shapes = load_shapes(name, datadir=DATA_DIR_CK,
                         resizeratio=RESIZERATIO)
    labels = load_labels

    dataset = to_dataset(aam.wparams, imgs, shapes)
    labels = load_labels(name, datadir=DATA_DIR_CK)
    mask = aam.wparams.warp_map
    return dataset, labels, mask
end


function save_images{T,N}(imgs::Vector{Array{T,N}}, path::UTF8String; prefix="img")
    for i=1:length(imgs)
        img = normalize(imgs[i])
        imwrite(img, joinpath(path, @sprintf("%s_%03d.png", prefix, i)))
    end
end

function save_comps{T}(X::Matrix{T}, mask::Matrix{Int}, path::UTF8String)
    imgs = [to_image(X[:, i], mask) for i=1:size(X, 2)]
    save_images(imgs, path)
end

distance(x, y) = sum(abs(x .- y))

function distance_matrix{T}(X::Matrix{T})
    d, n = size(X)
    D = zeros(Float64, n, n)
    for j=1:n
        for i=1:n
            D[i, j] = distance(X[:, i], X[:, j])
        end
        println("column $j")
    end
    return D
end

function dissimilar(X)
    D = distance_matrix(X)
    R = dbscan(D, 50, 1)
    return X[:, R.seeds]
end


# 1993 - S55
# 1990+1700 - S74

## function train_and_save_aam_ck()
##     imgs_aam = load_images(:ck, datadir=DATA_DIR_CK,
##                            resizeratio=RESIZERATIO, start=2000, count=2000)
##     shapes_aam = load_shapes(:ck, datadir=DATA_DIR_CK,
##                              resizeratio=RESIZERATIO, start=2000, count=2000)
##     @time aam = train(AAModel(), imgs_aam, shapes_aam);
##     save(joinpath(DATA_DIR_CK, "aam.jld"), "aam", aam)
## end

function train_and_save_aam_put()
    imgs = collect(Array{Float64,3},
                       load_images(FaceDatasets.PutFrontalDataset,
                                   joinpath(DATA_DIR, "PUT")));
    shapes = collect(Array{Float64,2},
                         load_shapes(FaceDatasets.PutFrontalDataset,
                                     joinpath(DATA_DIR, "PUT")));
    @time m = train(AAModel(), imgs_aam, shapes_aam)
    save(joinpath(joinpath(DATA_DIR, "models"), "aam_put.jld"), "aam", m)

    for i=1:10
        try
            img_idx = rand(1:length(imgs))
            shape_idx = rand(1:length(imgs))        
            triplot(imgs[img_idx], shapes[shape_idx], m.wparams.trigs)
            @time fitted_shape, fitted_app = fit(m, imgs[img_idx], shapes[shape_idx], 30);
            triplot(imgs[img_idx], fitted_shape, m.wparams.trigs)
            println("Image #$img_idx; shape #$shape_idx")
            # readline(STDIN)
        catch e
            if isa(e, BoundsError)
                println("Fitting diverged")
                # readline(STDIN)
            else
            rethrow()
            end
        end
    end
end


function load_aam()
    return load(joinpath(DATA_DIR_CK, "aam.jld"))["aam"]
end


function save_dataset(dataset)
    h5write(joinpath(DATA_DIR_CK, "dataset_10708.h5"), "dataset", dataset)
end

read_dataset() = h5read(joinpath(DATA_DIR_CK, "dataset.h5"), "dataset")


function test_projections()
    P = projection(fit(PCA, dataset))
    nview(to_image(P[:, 1], aam.wparams.warp_map))
    pca_imgs = Matrix{Float64}[normalize(to_image(P[:, i],
                                                  aam.wparams.warp_map))
                               for i=1:size(P, 2)]
    save_images(pca_imgs,
                expanduser("~/Dropbox/PhD/MyPapers/facial_expr_repr/images/pca"))
end


function view_and_save_rbm()
    comps = components(rbm)
    rbm_imgs = [to_image(comps[:, i], mask) for i=1:size(comps, 2)]
    nviewall(rbm_imgs[1:36])

    save_images(convert(Vector{Matrix{Float64}}, rbm_imgs),
                expanduser("~/Dropbox/PhD/MyPapers/facial_expr_repr/images/rbm"))

    h5open(joinpath(DATA_DIR_CK, "rbm_1.h5"), "w") do h5
        save_params(h5, rbm, "rbm")
    end

    h5open(joinpath(DATA_DIR_CK, "rbm_1.h5")) do h5
        load_params(h5, rbm, "rbm")
    end
end


function save_images(dir::AbstractString, imgs::Vector;
                     prefix="img", suffix=".png")
    for (i, img) in enumerate(imgs)
        save(joinpath(dir, prefix * string(i) * suffix), img)
    end
    println("Wrote $(length(imgs)) images to $dir")
end


