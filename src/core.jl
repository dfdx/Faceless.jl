
using PiecewiseAffineTransforms
using MultivariateStats
using ActiveAppearanceModels
using Images
using ImageView
using Colors
using FaceDatasets
using Boltzmann
using HDF5
using JLD
using Clustering

include("view.jl")


const DATA_DIR_CK = expanduser("~/data/CK")
const RESIZERATIO = .5


function to_dataset(wparams::PAWarpParams, imgs::Vector{Matrix{Float64}},
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


nview(img) = ImageView.view(normalize(img))


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


function train_and_save_aam()
    imgs_aam = load_images(:ck, datadir=DATA_DIR_CK,
                           resizeratio=RESIZERATIO, start=2000, count=2000)
    shapes_aam = load_shapes(:ck, datadir=DATA_DIR_CK,
                             resizeratio=RESIZERATIO, start=2000, count=2000)
    @time aam = train(AAModel(), imgs_aam, shapes_aam);
    save(joinpath(DATA_DIR_CK, "aam.jld"), "aam", aam)
end

function load_aam()
    return load(joinpath(DATA_DIR_CK, "aam.jld"))["aam"]
end


function save_dataset(dataset)
    h5write(joinpath(DATA_DIR_CK, "dataset.h5"), "dataset", dataset)
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



## function run_shapes()
##     shapes = load_shapes(:ck, datadir=DATA_DIR_CK, resizeratio=.5)

##     shape_data = hcat([reshape(s, length(s)) for s in shapes]...)
##     X = normalize(shape_data)
        
##     rbm = GRBM(size(X, 1), 6)
##     @time fit(rbm, X, n_gibbs=3, lr=0.1, n_iter=1000)

##     W = normalize(rbm.W') * 300
##     viewshape(zeros(300, 400), reshape(W[:, 2], 68 ,2))
## end




function main()
    aam = load_aam()

    # idxs = sample(1:count, count, replace=false)
    imgs = load_images(:ck, datadir=DATA_DIR_CK,
                       resizeratio=RESIZERATIO)
    shapes = load_shapes(:ck, datadir=DATA_DIR_CK,
                         resizeratio=RESIZERATIO)
    
    dataset = to_dataset(aam.wparams, imgs, shapes)
    mask = aam.wparams.warp_map
    imgs = nothing

    # nviewall([to_image(dataset[:, i], mask) for i=1:36])

    rbms = Array(Any, 3)
    for (i, p) in enumerate([1, 3, 5])
        println("-------- parameter = $p -----------")
        rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.001)
        @time fit(rbm, dataset, n_gibbs=p, lr=0.001,
                  batch_size=1000, n_epochs=100)
        nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
        # nviewall([to_image(rbm.W'[:, i], mask) for i=37+36:72+46])
        rbms[i] = rbm
    end
    readline(STDIN)
    
end


# Boltzmann.jl commit 343fce064c162a62215af7c1be07eea5b5bc2762
# resize ratio = 0.5 (~12k visible variables)

# Experiment 1: all good, but only 2 different faces (similar as well)
# more gibbs steps gives slightly better results
    ## for n_gibbs in [1, 3, 5]
    ##     println("-------- n_gibbs = $n_gibbs -----------")
    ##     rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.001)
    ##     @time fit(rbm, dataset, n_gibbs=n_gibbs, lr=0.01,
    ##               batch_size=1000, n_epochs=100)
    ##     nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
    ## end


# Experiment 2: sigma=0.1 - high distortion, sigma=0.01 - medium distortion, sigma=0.001 - no distortion
# all very similar
    ## for sigma in [0.001, 0.01, 0.1]
    ##     println("-------- sigma = $sigma -----------")
    ##     rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=sigma)
    ##     @time fit(rbm, dataset, n_gibbs=3, lr=0.01,
    ##               batch_size=1000, n_epochs=100)
    ##     nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
    ## end


# Experiment 3: sigma=0.5 - total shit, 100% distorsion
##         rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.5)
##         @time fit(rbm, dataset, n_gibbs=3, lr=0.01,
##                   batch_size=1000, n_epochs=100)
##         nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])


# Experiment 4: lr=0.001 and lr=0.01 - almost useless (pure noise), lr=0.1 and lr=1 - better (or vice versa?)
    ## for lr in [0.001, 0.01, 0.1, 1.]
    ##     println("-------- lr = $lr -----------")
    ##     rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.1)
    ##     @time fit(rbm, dataset, n_gibbs=3, lr=lr,
    ##               batch_size=1000, n_epochs=100)
    ##     nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
    ## end

# !!! Experiment 5: many similar, but in general quite good
## rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.001)
## @time fit(rbm, dataset, n_gibbs=3, lr=0.001,
##           batch_size=1000, n_epochs=100)
## nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])


# Experiment 6: n_gibbs > 1 doesn't make any difference
## for (i, p) in enumerate([1, 3, 5])
##         println("-------- parameter = $p -----------")
##         rbm = RBM(Degenerate, Bernoulli, size(dataset, 1), 1024, sigma=0.001)
##         @time fit(rbm, dataset, n_gibbs=p, lr=0.001,
##                   batch_size=1000, n_epochs=100)
##         nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
##         # nviewall([to_image(rbm.W'[:, i], mask) for i=37:72])
##         rbms[i] = rbm
##     end



# pseudo-likelihood progress (number of iterations - likelihood)
# 40 - 9471
# 50 - 9426

# some results from previous experiments
# n_hid=192, n_gibbs=3, n_meta=10 ~ 7 faces of good quality
# n_hid=192, n_gibbs=5, n_meta=10 ~ 4 faces of good quality
# n_hid=256, n_gibbs=1, n_meta=10 ~ no faces at all
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=10, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=20 ~ 12 faces of almost good quality
# n_hid=128, n_gibbs=3, n_meta=10 ~ all faces of bad quality
# n_hid=192, n_gibbs=3, n_meta=10 ~ 3 faces of a modate quality
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of moderate quality
# n_hid=320, n_gibbs=3, n_meta=10 ~ 16 faces of almost good qualiy
# n_hid=384, n_gibbs=3, n_meta=10 ~ 17 faces of almost good qualiy
# n_hid=512, n_gibbs=3, n_meta=10 ~ 22 faces of different quality
# n_hid=768, n_gibbs=3, n_meta=10 ~ 25 faces of different quality
# n_hid=1024, n_gibbs=3, n_meta=10 ~ 21 face of different quality
# n_hid=768, n_gibbs=3, n_meta=10, sigma=0.001 ~ almost all faces!
# n_hid=512, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good, but many similar
# n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good
# n_hid=128, n_gibbs=3, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# n_hid=128, n_gibbs=10, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# brbm: n_hid=768, n_gibbs=3, n_meta=10, lr=0.01 ~ 20 good
# grbm: n_hid=1024, n_gibbs=3, n_meta=10 ~ half almost good
# grbm: n_hid=768, n_gibbs=3, n_meta=10 ~ half almost good
# grbm: n_hid=1024, n_gibbs=10, n_meta=10 ~ half almost good
# grbm: n_hid=1024, n_gibbs=10, n_meta=3, lr=0.01 ~ half almost good
#! grbm: n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001, lr=0.01 ~ all really good!
# brbm: n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001, lr=0.01 ~ almost the same

# grbm (w/ momentum): n_hid=1024, n_gibbs=2, n_meta=5, sigma=0.001, lr=0.01 ~ 2243
