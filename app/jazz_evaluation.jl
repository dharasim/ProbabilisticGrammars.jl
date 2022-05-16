include("jazz_models.jl")

using ProgressMeter: @showprogress

function naive_supervised_evaluation_accs(grammar, trees)
    @showprogress map(trees) do tree
        prediction = predict_tree(grammar, leaflabels(tree))
        tree_similarity(tree, prediction)
    end
end

trees = [tune["harmony_tree"] for tune in treebank];
grammar = train_on_trees!(simple_harmony_grammar(rulekinds=[:rightheaded]), trees);
mean(naive_supervised_evaluation_accs(grammar, trees))

trees = [tune["rhythm_tree"] for tune in treebank];
grammar = train_on_trees!(simple_rhythm_grammar(), trees);
mean(naive_supervised_evaluation_accs(grammar, trees))

trees = [tune["harmony_tree"] for tune in treebank];
grammar = train_on_trees!(transpinv_harmony_grammar(rulekinds=[:rightheaded]), trees);
mean(naive_supervised_evaluation_accs(grammar, trees))

trees = [tune["product_tree"] for tune in treebank];
grammar = train_on_trees!(simple_product_grammar(rulekinds=[:rightheaded]), trees);
mean(naive_supervised_evaluation_accs(grammar, trees))

