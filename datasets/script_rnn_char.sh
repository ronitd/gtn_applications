CONFIG="$HOME/gtn_applications/recipes/diff_wfst/interspeech/config_rnn_char.json"
CHECKPOINT="$HOME/gtn_applications/trained_models/rnn_ctc_char"
python ../test.py --config $CONFIG --checkpoint $CHECKPOINT --split "train"
