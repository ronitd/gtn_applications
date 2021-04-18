import gtn
import numpy as np
import math
#from IPython.display import display, Image
# import sys
# sys.path.append('./scripts/')
# import audioset
import scripts.load_arpa as arpa
import pywarpfst as fst
# from mfst import FST, RealSemiringWeight
import pickle


UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


class TrieNode(object):
    """
    Our trie node implementation. Very basic. but does the job
    """

    def __init__(self, char: str):
        self.char = char
        self.children = {}
        # Is it the last character of the word.`
        self.word_finished = False
        # How many times this character appeared in the addition process
        self.counter = 1
        self.word = ""


def add(root, phonemes, word):
    """
    Adding a word in the trie structure
    """
    node = root
    node.counter += 1
    for char in phonemes:
        found_in_child = False
        # Search for the character in the children of the present `node`
        if char in node.children:
            node.children[char].counter += 1
            found_in_child = True
            node = node.children[char]
        # We did not find it so add a new chlid
        if not found_in_child:
            new_node = TrieNode(char)
            node.children[char] = new_node
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a word.
    node.word_finished = True
    node.word = word


def language_graph(tokens_to_idx, root, vocab, blank_idx):
    # g = gtn.Graph(False)
    g = fst.VectorFst()
    epsilon=0
    q, count_q = [], []
    q.append(root)
    count = 0
    count_q.append(count)

    end = 1
    count = 1
    s = g.add_state()
    g.set_state(s)
    f = g.add_state()
    g.set_final(f, 1)
    while q:
        curr = q.pop(0)
        curr_count = count_q.pop(0)
        for child in curr.children:
            count += 1
            g.add_state()
            # curr.children[child].counter / curr.counter

            g.add_arc(curr_count, fst.Arc(ilabel=tokens_to_idx[curr.children[child].char]+1, olabel=epsilon,
                      weight=curr.children[child].counter / curr.counter, nextstate=count)
                      )
            g.add_arc(count, fst.Arc(ilabel=blank_idx + 1, olabel=epsilon, weight=0, nextstate=count))

            g.add_arc(count, fst.Arc(ilabel=tokens_to_idx[curr.children[child].char] + 1, olabel=epsilon, weight=0,
                                     nextstate=count))
            count_q.append(count)
            q.append(curr.children[child])
        if curr.word_finished:

            g.add_arc(curr_count, fst.Arc(ilabel=epsilon, olabel=vocab[curr.word]+1, weight=0, nextstate=end))
            g.add_arc(curr_count, fst.Arc(ilabel=tokens_to_idx['▁'], olabel=vocab[curr.word] + 1, weight=0,
                                          nextstate=end))

    g = g.closure()
    g.arcsort(sort_type="olabel")
    g.write("L.fst")
    # pickle.dump(g, file=open('L', 'wb'))
    # g.arc_sort(olabel=True)
    # gtn.savetxt('L.txt', g)
    # g = gtn.loadtxt('L.txt')
    return g


def build_lm_graph(ngram_counts, vocab):
    graph = fst.VectorFst()
    lm_order = len(ngram_counts)
    assert lm_order > 1, "build_lm_graph doesn't work for unigram LMs"
    state_to_node = {}
    epsilon = 0

    def get_node(state):
        node = state_to_node.get(state, None)
        if node is not None:
            return node
        # is_start = state == tuple([vocab[BOS]])
        # is_end = vocab[EOS] in state
        node = graph.add_state()
        if state == tuple([vocab[BOS]]):
            graph.set_start(node)
        if vocab[EOS] in state:
            graph.set_final(node, 1)
        state_to_node[state] = node
        return node

    for counts in ngram_counts:
        for ngram in counts.keys():
            istate, ostate = ngram[0:-1], ngram[1 - lm_order:]
            inode = get_node(istate)
            onode = get_node(ostate)
            prob, bckoff = counts[ngram]
            # p(gram[-1] | gram[:-1])
            lbl = ngram[-1] if ngram[-1] != vocab[EOS] else epsilon

            graph.add_arc(inode, fst.Arc(ilabel=lbl, olabel=lbl, weight=prob, nextstate=onode))
            if bckoff is not None and vocab[EOS] not in ngram:
                bnode = get_node(ngram[1:])
                graph.add_arc(onode, fst.Arc(ilabel=epsilon, olabel=epsilon, weight=bckoff, nextstate=bnode))
    graph.arcsort(sort_type="ilabel")
    graph.write("G.fst")
    return graph


def build_lexicon_graph(lexicon_path):
    with open(lexicon_path, "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        epsilon=0
        g = fst.VectorFst()
        s = g.add_state()
        g.set_start(s)
        f = g.add_state()
        g.set_final(f)
        for l in lexicon:
            prev = s
            for i, val in enumerate(l[1:]):
                curr = g.add_state()
                olabel = l[0] if i==0 else epsilon
                g.add_arc(prev, fst.Arc(ilabel=val, olabel=olabel, weight=0, nextstate=curr))
                prev = curr
            g.add_arc(prev, fst.Arc(ilabel=val, olabel=l[0], weight=0, nextstate=f))
        g.add_arc(f, fst.Arc(ilabel=epsilon, olabel=epsilon, weight=0, nextstate=s))
    print("L Done")
    print(g.num_states())
    g.arcsort(sort_type="olabel")
    g.write("L-Si.fst")
    return g


def compose_language_grammar(tokens_to_index, lexicon_path, lm_path):
    counts, vocab = arpa.read_counts_from_arpa(lm_path)
    # symb = {v: k for k, v in vocab.items()}
    # root = TrieNode('*')
    # d = {}
    # with open(lexicon_path, "r") as fid:
    #     lexicon = (l.strip().split() for l in fid)
    #     for l in lexicon:
    #         d[l[0]] = l[1:]
    # for word in vocab:
    #     if word in d:
    #         add(root, d[word], word)
    #
    # g_lexicon = language_graph(tokens_to_index, root, vocab, len(tokens_to_index))
    g_lexicon= build_lexicon_graph(lexicon_path)
    print("Lexicon Done")
    g_lm = build_lm_graph(counts, vocab)
    print('G Done')
    LG = fst.compose(g_lexicon, g_lm)
    LG.write("LG.fst")

    print('Both Done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute data stats.")
    parser.add_argument("--lexicon_path", type=str, help="Path to dataset JSON files.",
                        default="/home/rjd2551/Speech/Gujarati/gtn/gujarati_dictionary_IITM_CommonLabelSet_final.txt")
    parser.add_argument(
        "--lm_arpa_path", type=str, help="Path to language model arpa file.",
        default="/home/rjd2551/Speech/Gujarati/gu-words.arpa"
    )
    parser.add_argument(
        "--tokens_path", type=str, help="Path to save tokens.", default=None
    )
    args = parser.parse_args()
    tokens_to_index = {'a': 0, 'aa': 1, 'ae': 2, 'ax': 3, 'b': 4, 'bh': 5, 'c': 6, 'ch': 7, 'd': 8, 'dh': 9, 'dx': 10,
                       'dxh': 11, 'ee': 12, 'ei': 13, 'g': 14, 'gh': 15, 'h': 16, 'hq': 17, 'i': 18, 'ii': 19, 'j': 20,
                       'jh': 21, 'k': 22, 'kh': 23, 'l': 24, 'lx': 25, 'm': 26, 'mq': 27, 'n': 28, 'ng': 29, 'nj': 30,
                       'nx': 31, 'o': 32, 'ou': 33, 'p': 34, 'ph': 35, 'q': 36, 'r': 37, 'rq': 38, 's': 39, 'sh': 40,
                       'sx': 41, 't': 42, 'th': 43, 'tx': 44, 'txh': 45, 'u': 46, 'uu': 47, 'w': 48, 'y': 49, '▁': 50}
    compose_language_grammar(tokens_to_index, args.lexicon_path, args.lm_arpa_path)

