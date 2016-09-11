
function ck_generate_test()
    X, y, mask = load_image_dataset(:ck_max)
    rbm = RBM(Degenerate, Bernoulli, 1, 1)
    h5open(joinpath(MODEL_DIR, "rbm_512.h5")) do h5
        load_params(h5, rbm, "model")
    end
    Xg = generate(rbm, X)
    imgs = [to_image(Xg[:,i], mask) for i in 1:size(Xg, 2)]
    nviewall(imgs[1:36])
end


function lfw_rbm()
    imgs = 
end
