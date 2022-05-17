include("jazz_models.jl")

begin
    using ProgressMeter: @showprogress
    using Random: randperm, default_rng
    using Base.Iterators: flatten
    using Logging: Logging
    Logging.disable_logging(Logging.Info)

    function prediction_accs(grammar, trees)
        map(trees) do tree
            prediction = predict_tree(grammar, leaflabels(tree))
            tree_similarity(tree, prediction)
        end
    end

    ## k-fold cross validation for n data points
    function cross_validation_index_split(num_folds, num_total, rng=default_rng())
    num_perfold = ceil(Int, num_total/num_folds)
    num_lastfold = num_total - (num_folds-1) * num_perfold
    fold_lenghts = [fill(num_perfold, num_folds-1); num_lastfold]
    fold_ends = accumulate(+, fold_lenghts)
    fold_starts = fold_ends - fold_lenghts .+ 1
    shuffled_idxs = randperm(rng, num_total)
    test_indices = [shuffled_idxs[i:j] for (i,j) in zip(fold_starts,fold_ends)]
    train_indices = [setdiff(1:num_total, idxs) for idxs in test_indices]
    return collect(zip(test_indices, train_indices))
    end

    function prediction_accs_crossval(treebankgrammar, trees, num_folds)
        index_splits = cross_validation_index_split(num_folds, length(trees))
        accss = map(index_splits) do (test_idxs, train_idxs)
            grammar = treebankgrammar(trees[train_idxs])
            prediction_accs(grammar, trees[test_idxs])
        end
        collect(flatten(accss))
    end
end



trees = [tune["harmony_tree"] for tune in treebank];
treebankgrammar(trees) = 
    use_map_params(train_on_trees!(simple_harmony_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    simple_harmony_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(grammar, trees))



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

for a in 0.70:0.01:0.85
    grammar = runvi(leaflabels.(trees), epochs=5, showprogress=false) do 
        regularized_transpinv_grammar(rulekinds=[:rightheaded], lvlaccept=a)
    end;
    m = 
    println(a, " | ", mean(prediction_accs(use_map_params(grammar), trees)))
end

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