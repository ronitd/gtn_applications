import gtn
import numpy as np
import math
#from IPython.display import display, Image
# import sys
# sys.path.append('./scripts/')
# import audioset
import scripts.load_arpa as arpa

from mfst import FST, RealSemiringWeight
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
    g = FST()
    epsilon=0
    q, count_q = [], []
    q.append(root)
    count = 0
    count_q.append(count)

    end = 1
    count = 1
    s = g.add_state()
    g.set_initial_state(s)
    f = g.add_state()
    g.set_final_weight(f, 1)
    while q:
        curr = q.pop(0)
        curr_count = count_q.pop(0)
        for child in curr.children:
            count += 1
            g.add_state()
            # curr.children[child].counter / curr.counter
            g.add_arc(curr_count, count, input_label=tokens_to_idx[curr.children[child].char]+1, output_label=epsilon,
                      weight=curr.children[child].counter / curr.counter
                      )
            g.add_arc(count, count, input_label=blank_idx + 1, output_label=epsilon, weight=0)
            g.add_arc(count, count, input_label=tokens_to_idx[curr.children[child].char] + 1, output_label=epsilon, weight=0)
            count_q.append(count)
            q.append(curr.children[child])
        if curr.word_finished:
            g.add_arc(curr_count, end, input_label=epsilon, output_label=vocab[curr.word]+1, weight=0)
            g.add_arc(curr_count, end, input_label=tokens_to_idx['▁'], output_label=vocab[curr.word]+1, weight=0)

    g = g.closure()
    pickle.dump(g, file=open('L', 'wb'))
    # g.arc_sort(olabel=True)
    # gtn.savetxt('L.txt', g)
    # g = gtn.loadtxt('L.txt')
    return g


def unigram(vocab, count, symb):
    unigram = []
    for word in vocab:
        g = gtn.Graph(False)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, vocab[word], vocab[word], count[0][(vocab[word],)][0])
        unigram.append(g)
    g = gtn.closure(gtn.union(unigram)).arc_sort()
    gtn.savetxt('U.txt', g)


def build_lm_graph(ngram_counts, vocab):
    graph = FST()
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
            graph.set_initial_state(node)
        if vocab[EOS] in state:
            graph.set_final_weight(node, 1)
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
            graph.add_arc(inode, onode, input_label=lbl, output_label=lbl, weight=prob)
            if bckoff is not None and vocab[EOS] not in ngram:
                bnode = get_node(ngram[1:])
                graph.add_arc(onode, bnode, input_label=epsilon, output_label=epsilon, weight=bckoff)

    return graph


def compose_language_grammar(tokens_to_index, lexicon_path, lm_path):
    counts, vocab = arpa.read_counts_from_arpa(lm_path)
    symb = {v: k for k, v in vocab.items()}
    root = TrieNode('*')
    d = {}
    with open(lexicon_path, "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        for l in lexicon:
            d[l[0]] = l[1:]
    for word in vocab:
        if word in d:
            add(root, d[word], word)

    g_lexicon = language_graph(tokens_to_index, root, vocab, len(tokens_to_index))
    print("Lexicon Done")
    g_lm = build_lm_graph(counts, vocab)
    # g_lm.arc_sort()
    # gtn.savetxt('G.txt', g_lm)
    pickle.dump(g_lm, file=open('G', 'wb'))
    print('G Done')
    LG = FST.compose(g_lexicon, g_lm)
    pickle.dump(LG, file=open('LG', 'wb'))
    # g.arc_sort()
    #gtn.savetxt('LG.txt', LG)
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

