### Structure

- `avg_recall_replication_exp.py` - script to plot Recall@k and Mean Rank values in the experiments
5.1 and 5.2 in the thesis (first two subsections in results section)
- `constants.py` - constants for the plots
- `get_avg_num_words.py` - calculates the average number of words in the sentence in the dataset we use, and 
plots barplots and boxplots about it
- `plot_Brittleness.py` - plots Brittleness
- `plot_distr_num_hard_sentences_per_caption.py` - plots the statistics of how many hard negatives or positives we
generated for each sentence using POS methodology (details in the paper, in the Methodology section)
- `plot_distr_num_hard_sentences_per_caption.py` - plots the distributions of hard positive/negative sentences per
caption; you can see the plots created by that in the appendix
- `plot_PosRank.py` - plots PosRank
- `plot_Recallk_MeanR.py` - plots Recall@k and Mean Rank for everything 
after first two experiments
