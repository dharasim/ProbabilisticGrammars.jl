using Distributed: addprocs, workers, @everywhere, CachingPool, pmap
addprocs(5; exeflags="--project")

@everywhere include("jazz_models.jl")
using .JazzModels
@everywhere const JM = JazzModels

@everywhere begin 
    using Logging: Logging
    Logging.disable_logging(Logging.Info)
end

using DataFrames: DataFrames, DataFrame

@everywhere begin 
    function model_trees(model)
        treekey = if model.rhythm == "none"
            "harmony_tree"
        elseif model.harmony == "none"
            "rhythm_tree"
        else
            "product_tree"
        end
        [tune[treekey] for tune in JM.treebank]
    end

    function rulekinds(model)
        if model.headedness == "left"
            [:leftheaded]
        elseif model.headedness == "right"
            [:rightheaded]
        elseif model.headedness == "either"
            [:leftheaded, :rightheaded]
        else
            error("$(model.headedness) is not a valid headedness")
        end
    end

    function mk_grammar(model)
        if     model.harmony == "simple" && model.rhythm == "none"
            JM.simple_harmony_grammar(rulekinds=rulekinds(model))
        elseif model.harmony == "transpinv" && model.rhythm == "none"
            JM.transpinv_harmony_grammar(rulekinds=rulekinds(model))
        #
        elseif model.harmony == "none" && model.rhythm == "simple"
            JM.simple_rhythm_grammar()
        elseif model.harmony == "none" && model.rhythm == "regularized"
            JM.regularized_rhythm_grammar()
        #
        elseif model.harmony == "simple" && model.rhythm == "simple"
            JM.simple_product_grammar(rulekinds=rulekinds(model))
        elseif model.harmony == "transpinv" && model.rhythm == "simple"
            JM.transpinv_product_grammar(rulekinds=rulekinds(model))
        elseif model.harmony == "simple" && model.rhythm == "regularized"
            JM.regularized_product_grammar(rulekinds=rulekinds(model))
        elseif model.harmony == "transpinv" && model.rhythm == "regularized"
            JM.regularized_transpinv_grammar(rulekinds=rulekinds(model))
        else
            error("no grammar implemented for model $model")
        end
    end

    function supervised_accs(model, index_splits; workers=workers())
        wp = CachingPool(workers)
        accs = pmap(wp, enumerate(index_splits)) do (foldid, (test_idxs, train_idxs))
            train_trees = model_trees(model)[train_idxs]
            grammar = JM.use_map_params(JM.train_on_trees!(mk_grammar(model), train_trees))
            accs = JM.prediction_accs(grammar, model_trees(model)[test_idxs])
            mode = "supervised"
            [(; mode, treeid, foldid, acc) for (treeid, acc) in zip(test_idxs, accs)]
        end
        reduce(vcat, accs)
    end

    function unsupervised_accs(model, epochs, index_splits; workers=workers())
        wp = CachingPool(workers)
        accs = pmap(wp, enumerate(index_splits)) do (foldid, (test_idxs, train_idxs))
            train_trees = model_trees(model)[train_idxs]
            grammar = JM.runvi(() -> mk_grammar(model), JM.leaflabels.(train_trees); epochs, showprogress=false)
            accs = JM.prediction_accs(grammar, model_trees(model)[test_idxs])
            mode = "unsupervised"
            [(; mode, treeid, foldid, acc) for (treeid, acc) in zip(test_idxs, accs)]
        end
        reduce(vcat, accs)
    end
end

model(harmony, rhythm, headedness) = (; harmony, rhythm, headedness)
model(spec) = model(spec...)
models = map(model, [
    ("simple", "none", "right"),
    ("simple", "none", "left"),
    # ("simple", "none", "either"),
    #
    ("transpinv", "none", "right"),
    ("transpinv", "none", "left"),
    # ("transpinv", "none", "either"),
    #
    ("none", "simple", "none"),
    ("none", "regularized", "none"),
    # 
    ("simple", "simple", "right"),
    ("simple", "simple", "left"),
    # ("simple", "simple", "either"),
    #
    ("transpinv", "simple", "right"),
    ("transpinv", "simple", "left"),
    # ("transpinv", "simple", "either"),
    # 
    ("simple", "regularized", "right"),
    ("simple", "regularized", "left"),
    # ("simple", "regularized", "either"),
    #
    ("transpinv", "regularized", "right"),
    ("transpinv", "regularized", "left"),
    # ("transpinv", "regularized", "either"),
])

num_runs = 2
num_epochs = 3
num_trees = 150
num_folds = 10

JM.cross_validation_index_split(num_folds, num_trees)

using Dates: Dates

crossval_data = DataFrame(
    mapreduce(vcat, 1:num_runs) do runid
        @show runid
        index_splits = JM.cross_validation_index_split(num_folds, num_trees)
        mapreduce(vcat, models) do model
            time = Dates.format(Dates.now(), "HH:MM:SS") 
            @show model, time
            vcat(
                map(supervised_accs(model, index_splits)) do accs
                    (; runid, model..., accs...)
                end,
                map(unsupervised_accs(model, num_epochs, index_splits)) do accs
                    (; runid, model..., accs...)
                end,
            )
        end
    end
)[:, [:harmony, :rhythm, :headedness, :mode, :runid, :treeid, :foldid, :acc]]

using CSV: CSV
CSV.write("app/crossval_data.csv", crossval_data)

final_time = Dates.format(Dates.now(), "HH:MM:SS") 
@show final_time



# let model = model("simple", "simple", "either")
#     train_trees = model_trees(model)
#     grammar = JM.runvi(() -> mk_grammar(model), JM.leaflabels.(train_trees); epochs=1, showprogress=false);
#     tic = time_ns()
#     accs = JM.prediction_accs(grammar, model_trees(model))
#     toc = time_ns()
#     (toc - tic) / 1_000_000_000
# end
# 
# 
# let model = model("simple", "simple", "either")
#     index_splits = JM.cross_validation_index_split(num_folds, num_trees)
#     wp = CachingPool(workers())
#     tic = time_ns()
#     pmap(wp, enumerate(index_splits)) do (foldid, (test_idxs, train_idxs))
#         train_trees = model_trees(model)[train_idxs]
#         grammar = JM.runvi(() -> mk_grammar(model), JM.leaflabels.(train_trees); epochs=1, showprogress=false)
#         accs = JM.prediction_accs(grammar, model_trees(model)[test_idxs])
#         # mode = "unsupervised"
#         # [(; mode, treeid, foldid, acc) for (treeid, acc) in zip(test_idxs, accs)]
#     end
#     toc = time_ns()
#     (toc - tic) / 1_000_000_000
# end
# 
# 
# 
# 
# 

trees = [tune["harmony_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(simple_harmony_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    simple_harmony_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(use_map_params(grammar), trees))



trees = [tune["rhythm_tree"] for tune in treebank];
treebankgrammar(trees) = use_map_params(train_on_trees!(simple_rhythm_grammar(), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    simple_rhythm_grammar()
end;
mean(prediction_accs(grammar, trees))



trees = [tune["rhythm_tree"] for tune in treebank];
treebankgrammar(trees) = use_map_params(train_on_trees!(regularized_rhythm_grammar(), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    regularized_rhythm_grammar()
end;
mean(prediction_accs(grammar, trees))



trees = [tune["harmony_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(transpinv_harmony_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    transpinv_harmony_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(grammar, trees))



trees = [tune["product_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(simple_product_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    simple_product_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(grammar, trees))



trees = [tune["product_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(regularized_product_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=5) do 
    regularized_product_grammar(rulekinds=[:rightheaded], lvlaccept=0.78)
end;
mean(prediction_accs(grammar, trees))

# (5 epochs each)
# 0.75 | 0.6089
# 0.76 | 0.6122
# 0.77 | 0.6075
# 0.78 | 0.6034



trees = [tune["product_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(regularized_transpinv_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=5, showprogress=false) do 
    regularized_transpinv_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(use_map_params(grammar), trees))

# (5 epochs each)
# 0.70 | 0.6119014428476931
# 0.71 | 0.6126362760972321
# 0.72 | 0.6134465390913187
# 0.73 | 0.6113393284651328
# 0.74 | 0.613065266689885
# 0.75 | 0.6134145922273035
# 0.76 | 0.6099476816605353
# 0.77 | 0.6069282887847308
# 0.78 | 0.6172561787462729
# 0.79 | 0.6136732944273081
# 0.80 | 0.6142903579193715
# 0.81 | 0.6121356268977051
# 0.82 | 0.6104933267027732
# 0.83 | 0.6094690692265035
# 0.84 | 0.6069427208735422
# 0.85 | 0.6040496426256251


trees = [tune["product_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(transpinv_product_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    transpinv_product_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(grammar, trees))

