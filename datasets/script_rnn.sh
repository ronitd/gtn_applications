CONFIG="$HOME/gtn_applications/recipes/diff_wfst/interspeech/config_rnn.json"
CHECKPOINT="$HOME/gtn_applications/trained_models/rnn_ctc"
python ../test.py --config $CONFIG --checkpoint $CHECKPOINT
