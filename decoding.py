import gtn
import numpy as np
import math
import pickle
import pdb
#from IPython.display import display, Image
# import sys
# sys.path.append('./scripts/')
# import audioset
import scripts.load_arpa as arpa
from collections import defaultdict

def save_obj(obj, name):
    with open('./obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('./obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
    g = gtn.Graph(False)
    q, count_q = [], []
    q.append(root)
    count = 0
    count_q.append(count)

    end = 1
    count = 1
    g.add_node(True)
    g.add_node(False, True)
    while q:
        curr = q.pop(0)
        curr_count = count_q.pop(0)
        for child in curr.children:
            count += 1
            g.add_node()
            # curr.children[child].counter / curr.counter
            g.add_arc(curr_count, count, tokens_to_idx[curr.children[child].char], gtn.epsilon, math.log(curr.children[child].counter / curr.counter)
                      )
            g.add_arc(count, count, blank_idx, gtn.epsilon, 0)
            g.add_arc(count, count, tokens_to_idx[curr.children[child].char], gtn.epsilon, 0)
            count_q.append(count)
            q.append(curr.children[child])
        if curr.word_finished:
            g.add_arc(curr_count, end, gtn.epsilon, vocab[curr.word], 0)
            g.add_arc(curr_count, end, tokens_to_idx["▁"], vocab[curr.word], 0)

    g = gtn.closure(g)
    g.arc_sort(olabel=True)
    gtn.savetxt('gu-L-prob.txt', g)
    exit()
    # g = gtn.loadtxt('L.txt')
    return g


def build_lexicon_graph(tokens_to_idx, vocab, d, phoneme_to_word):
    epsilon = 0
    g = gtn.Graph(False)
    s = g.add_node(True)
    f = g.add_state(False, True)
    for word_idx, word in enumerate(vocab):
        prev = s
        if word in d and phoneme_to_word[tuple(d[word])][0] == word:
            for i, val in enumerate(d[word]):
                curr = g.add_node()
                oplabel = word_idx+1 if i == 0 else epsilon
                g.add_arc(prev, curr, tokens_to_idx[val]+1, oplabel, 0)
                prev = curr
            g.add_arc(prev, f, epsilon, epsilon, 0)
    g.add_arc(f, s, epsilon, epsilon, 0)
    print(g.num_nodes())

    g.arcsort(olabel=True)
    # g.write("L-Si.fst")
    #
    #
    # text_file = open("L-Si.txt", "w")
    # text_file.write(str(g))
    # text_file.close()
    #exit()
    gtn.savetxt("gu-L-gtn")
    return g


def compose_language_grammar(tokens_to_index, lexicon_path, lm_path):
    counts, vocab = arpa.read_counts_from_arpa(lm_path)
    
    symb = {v: k for k, v in vocab.items()}
    save_obj(symb, 'gu-tokens-to-word')
    #print("SymbL", symb)
    # exit()
    root = TrieNode('*')
    d = {}
    phoneme_to_word = defaultdict(list)
    with open(lexicon_path, "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        for l in lexicon:
            d[l[0]] = l[1:]
            phoneme_to_word[tuple(l[1:])].append(l[0])
    for word in vocab:
        if word in d:
            add(root, d[word], word)

    # g_lexicon = language_graph(tokens_to_index, root, vocab, len(tokens_to_index))
    g_lexicon = build_lexicon_graph(tokens_to_index, vocab, d, phoneme_to_word)
    g_lm = arpa.build_lm_graph(counts, vocab)
    g_lm.arc_sort()
    print("G nodes: ", g_lm.num_states())
    gtn.savetxt('gu-G-gtn.txt', g_lm)
    g = gtn.compose(g_lexicon, g_lm)
    g.arc_sort()

    g = gtn.compose(language_graph(tokens_to_index, root, vocab, len(tokens_to_index)), arpa.build_lm_graph(counts, vocab).arc_sort())
    print("LG states", g.num_nodes())
    gtn.savetxt('gu-LG-gtn.txt', g)
    # gtn.savetxt('gu-LG-gtn.txt', gtn.compose(language_graph(tokens_to_index, root, vocab, len(tokens_to_index)), arpa.build_lm_graph(counts, vocab).arc_sort()).arc_sort())


def unigram(vocab, count):
    print(len(vocab))
    unigram = []
    for word in vocab:
        g = gtn.Graph(False)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, vocab[word], vocab[word], count[0][(vocab[word],)][0])
        unigram.append(g)
    print(len(unigram))
    g = gtn.closure(gtn.union(unigram))
    g.arc_sort()
    gtn.savetxt('gu-U.txt', g)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute data stats.")
    parser.add_argument("--lexicon_path", type=str, help="Path to dataset JSON files.",
                        default="/home/rjd2551/Speech/Gujarati/gtn/gujarati_dictionary_IITM_CommonLabelSet_final.txt")
    parser.add_argument(
        "--lm_arpa_path", type=str, help="Path to language model arpa file.",
        default="/home/rjd2551/Speech/Gujarati/gu-bi-gram-word.arpa"
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
    '''
    L = gtn.loadtxt('gu-L.txt')
    print("L Done")
    G = gtn.loadtxt('gu-G.txt')
    print("G loaded")
    pdb.set_trace()
    LG = gtn.compose(L, G)
    LG.arc_sort()
    gtn.savetxt('gu-LG.txt',LG)
    '''
    #counts, vocab = arpa.read_counts_from_arpa(args.lm_arpa_path)
    #unigram(vocab, counts)  
    gtn.savetxt('gu-UG.txt', gtn.compose(gtn.loadtxt('gu-L.txt'), gtn.loadtxt('gu-U.txt')).arc_sort())


