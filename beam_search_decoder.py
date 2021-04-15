import torch


class BeamCTCDecoder:

    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40,
                 cutoff_prob=-2.1, beam_width=100, num_processes=4, blank_index=0, vocab=None, trie=None):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires ctcdecoder package")

        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob,
                                       beam_width=beam_width, num_processes=num_processes, blank_index=blank_index,
                                       log_probs_input=True)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = len(utt)
                # size = seq_len[p]
                # print("utt: {}".format(utt))
                # print("utt size: {}".format(utt.size))
                if size > 0:
                    transcript = "".join(
                        map(lambda x: self.int_to_char[x.item()] if x.item() in self.int_to_char.keys() else "",
                            utt[0:size]))
                else:
                    transcript = ""
                utterances.append(utterances)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def custom_convert_to_string(self, tokens, vocab, seq_lens):
        # return ''.join([vocab[x] for x in tokens[0:seq_len]])
        strings = []
        # print("Tokens shape: {}".format(tokens.shape))
        for i in range(len(tokens)):
            # Batches
            token = tokens[i][0]
            seq_len = seq_lens[i][0]
            # print("i = {} - Token shape: {} -- Seq len: {}".format(i, token.shape, seq_len))
            decoded_string = ''.join([vocab[x] for x in token[0:seq_len]])
            strings.append(decoded_string)
        return strings


    def decode(self, probs, sizes=None):
        # print("In decode")
        # print("Probs before pow: {}".format(probs))
        # probs = torch.pow(np.exp(1), probs).cpu()
        # print("Probs shape: {}".format(probs.shape))
        # print("Probs after pow: {}".format(probs))
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)
        # out, scores = self._decoder.decode(probs)
        # offsets = 0
        # print("After decode")
        # print("Seq lens shape: {}".format(seq_lens.shape))
        strings = self.convert_to_strings(out, seq_lens)
        # strings = self.custom_convert_to_string_ronit(out, self.labels, seq_lens=None)
        # offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets
