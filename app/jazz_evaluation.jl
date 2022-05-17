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
    use_map_params(train_on_trees!(transpinv_product_grammar(rulekinds=[:rightheaded]), trees))
mean(prediction_accs(treebankgrammar(trees), trees))
mean(prediction_accs_crossval(treebankgrammar, trees, 15))

grammar = runvi(leaflabels.(trees), epochs=3) do 
    transpinv_product_grammar(rulekinds=[:rightheaded])
end;
mean(prediction_accs(grammar, trees))