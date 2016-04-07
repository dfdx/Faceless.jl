
include("core.jl")


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

