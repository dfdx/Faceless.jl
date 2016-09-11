
include("core.jl")


function ck_train()
    aam = load_aam()
    
    # idxs = sample(1:count, count, replace=false)
    imgs = load_images(CKDataset, DATA_DIR_CK,
                       resizeratio=RESIZERATIO)
    shapes = load_shapes(CKDataset, DATA_DIR_CK,
                         resizeratio=RESIZERATIO)

    dataset = to_dataset(aam.wparams, imgs, shapes)
    mask = aam.wparams.warp_map
    imgs = nothing

    # nviewall([to_image(dataset[:, i], mask) for i=1:36])

    rbms = Array(Any, 8)
    for (i, p) in enumerate([512, 1024, 2048])
        println("-------- param1 = $p -----------")
        rbm = RBM(Degenerate, Bernoulli, size(dataset, 1),
                  p, sigma=0.01)
        @time fit(rbm, dataset, n_gibbs=1, lr=0.01,
                  batch_size=1000, n_epochs=20,
                  weight_decay_kind=:l2, weight_decay_rate=0.5,
                  sparsity_cost=0.1, sparsity_target=0.01)
        nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
        # nviewall([to_image(rbm.W'[:, i], mask) for i=37:72])
        rbms[i] = rbm
    end
    h5open(joinpath(MODEL_DIR, "rbm_512.h5"), "w") do h5
        save_params(h5, rbms[1], "model")
    end
    h5open(joinpath(MODEL_DIR, "rbm_1024.h5"), "w") do h5
        save_params(h5, rbms[2], "model")
    end
    h5open(joinpath(MODEL_DIR, "rbm_2048.h5"), "w") do h5
        save_params(h5, rbms[3], "model")
    end
end


function put_train()
    aam = load(joinpath(MODEL_DIR, "aam_put.jld"))["aam"]
    
    # idxs = sample(1:count, count, replace=false)
    imgs = collect(Array{Float64,3},
                   load_images(PutFrontalDataset, DATA_DIR_PUT))
    shapes = collect(Array{Float64,2},
                     load_shapes(PutFrontalDataset, DATA_DIR_PUT))

    dataset = to_dataset(aam.wparams, imgs, shapes)
    mask = aam.wparams.warp_map
    # imgs = nothing

    nviewall([to_image(dataset[:, i], mask)
              for i=rand(1:size(dataset,2), 9)])

    rbms = Array(Any, 8)
    for (i, p) in enumerate([3072])
        println("-------- param1 = $p -----------")
        rbm = RBM(Degenerate, Bernoulli, size(dataset, 1),
                  p, sigma=0.01)
        @time fit(rbm, dataset, n_gibbs=1, lr=0.01,
                  batch_size=1000, n_epochs=20,
                  weight_decay_kind=:l2, weight_decay_rate=0.5,
                  sparsity_cost=0.1, sparsity_target=0.01)
        nviewall([to_image(rbm.W'[:, i], mask) for i=1:36])
        # nviewall([to_image(rbm.W'[:, i], mask) for i=37:72])
        rbms[i] = rbm
    end
    h5open(joinpath(MODEL_DIR, "rbm_512.h5"), "w") do h5
        save_params(h5, rbms[1], "model")
    end
    h5open(joinpath(MODEL_DIR, "rbm_1024.h5"), "w") do h5
        save_params(h5, rbms[2], "model")
    end
    h5open(joinpath(MODEL_DIR, "rbm_2048.h5"), "w") do h5
        save_params(h5, rbms[3], "model")
    end
end


function kaggle_fer_train()
    imgs = load_images(:kaggle_fer, datadir=joinpath(DATA_DIR, "fer2013"))
    labels = load_lables(:kaggle_fer, datadir=joinpath(DATA_DIR, "fer2013"))
    dataset = hcat([reshape(img, 48*48) for img in imgs]...)
    rbm = RBM(Degenerate, Bernoulli, 48*48, 500)
    @time fit(rbm, dataset, lr=0.1, n_epochs=10,
              weight_decay_kind=:l2, weight_decay_rate=0.5)
              # sparsity_cost=0.1, sparsity_target=0.01)
    nviewall([reshape(rbm.W'[:, i], 48, 48) for i in 1:36])
end
