for np in 200 500 1000 1500
do
	python ../scripts/make_wordpieces.py \
		--dataset interspeech \
		--data_dir "/home/rjd2551/Speech/Gujarati/gtn/" \
		--output_prefix "/home/rjd2551/Speech/Gujarati/gtn/word_pieces" \
		--text_file "/home/rjd2551/Speech/Gujarati/gu-corpus.txt" \
		--num_pieces $np
done
