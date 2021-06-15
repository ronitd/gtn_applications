CONFIG="$HOME/gtn_applications/recipes/diff_wfst/interspeech/config_rnn_word_decomposition.json"
CHECKPOINT="$HOME/gtn_applications/trained_models/rnn_ctc_word_decoposition"
python ../test.py --config $CONFIG --checkpoint $CHECKPOINT 
