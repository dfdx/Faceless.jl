
using MLBase
# using LIBSVM
# using Mocha

include("core.jl")

function svm_simple()
    Xfull, yfull, mask = load_image_dataset(:ck_max)
    X = Xfull[:, yfull .!= -1]
    y = yfull[yfull .!= -1]
    # train_idxs = collect(StratifiedKfold(y, 10))[1]
    # test_idxs = setdiff(1:length(y), train_idxs)
    train_idxs = collect(1:200)
    test_idxs = collect(201:length(y))
    @time model = svmtrain(y[train_idxs], X[:, train_idxs])
    @time yhat = svmpredict(model, X[:, test_idxs])

    @printf "%.2f%%\n" mean(yhat .== y[test_idxs])*100
end


function svm_rbm()
    Xfull, yfull, mask = load_image_dataset(:ck_max)
    X = Xfull[:, yfull .!= -1]
    y = yfull[yfull .!= -1]
    # train_idxs = collect(StratifiedKfold(y, 10))[1]
    # test_idxs = setdiff(1:length(y), train_idxs)
    train_idxs = collect(1:200)
    test_idxs = collect(201:length(y))

    rbm = RBM(Degenerate, Bernoulli, 1, 1)
    h5open(joinpath(MODEL_DIR, "rbm_512.h5")) do h5
        load_params(h5, rbm, "model")
    end
    Xt = Boltzmann.transform(rbm, X)

    @time model = svmtrain(y[train_idxs], Xt[:, train_idxs])
    @time yhat = svmpredict(model, Xt[:, test_idxs])

    @printf "%.2f%%\n" mean(yhat .== y[test_idxs])*100
end


function mocha_simple()
    Xfull, yfull, mask = load_image_dataset(:ck_max)
    X = Xfull[:, yfull .!= -1]
    y = yfull[yfull .!= -1]
    # train_idxs = collect(StratifiedKfold(y, 10))[1]
    # test_idxs = setdiff(1:length(y), train_idxs)
    train_idxs = collect(1:200)
    test_idxs = collect(201:length(y))

    train_data_path = joinpath(MODEL_DIR, "train_data_simple.h5")
    test_data_path = joinpath(MODEL_DIR, "test_data_simple.h5")
    h5open(train_data_path, "w") do h5
        write(h5, "data", X[:, train_idxs])
        write(h5, "label", y[train_idxs])
    end
    h5open(test_data_path, "w") do h5
        write(h5, "data", X[:, test_idxs])
        write(h5, "label", y[test_idxs])
    end

    data  = HDF5DataLayer(name="train-data", source="train-data.txt",
                          batch_size=64)
    ## conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
    ## pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])
    ## conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
    ## pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])
    fc1   = InnerProductLayer(name="ip1",output_dim=500,neuron=Neurons.ReLU(),bottoms=[:data],
                              tops=[:ip1])
    ## fc2   = InnerProductLayer(name="ip2",output_dim=10,bottoms=[:ip1],tops=[:ip2])
    loss  = SoftmaxLossLayer(name="loss",bottoms=[:ip1,:label])

    backend = DefaultBackend()
    init(backend)

    net = Net("MNIST-train", backend, [data, fc1, loss])

    exp_dir = "snapshots"
    solver_method = SGD()
    params = make_solver_parameters(solver_method, max_iter=10000, regu_coef=0.0005,
                                    mom_policy=MomPolicy.Fixed(0.9),
                                    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                    load_from=exp_dir)
    solver = Solver(solver_method, params)

    setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

    # report training progress every 100 iterations
    add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

    # save snapshots every 5000 iterations
    add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

    # show performance on test data every 1000 iterations
    data_test = HDF5DataLayer(name="test-data",source="test-data.txt",batch_size=100)
    accuracy = AccuracyLayer(name="test-accuracy",bottoms=[:ip1, :label])
    test_net = Net("MNIST-test", backend, [data_test, fc1, accuracy])
    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

    solve(solver, net)

    destroy(net)
    destroy(test_net)

end

