import gtn
import numpy as np
import math
from IPython.display import display, Image
# import sys
# sys.path.append('./scripts/')
import scripts.load_arpa as arpa

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
    d = {0: '*', -2: "_"}

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
            g.add_arc(curr_count, count, tokens_to_idx[curr.children[child].char], gtn.epsilon,
                      curr.children[child].counter / curr.counter)
            g.add_arc(count, count, blank_idx, gtn.epsilon, 0)
            g.add_arc(count, count, tokens_to_idx[curr.children[child].char], gtn.epsilon, 0)
            count_q.append(count)
            q.append(curr.children[child])
            d[count] = child
        if curr.word_finished:
            g.add_arc(curr_count, end, gtn.epsilon, vocab[curr.word], 0)
            g.add_arc(curr_count, end, tokens_to_idx['_'], vocab[curr.word], 0)

    g = gtn.closure(g)

    gtn.savetxt('L.txt', g)
    # g = gtn.loadtxt('L.txt')
    return g


def compose_language_grammer(tokens_to_index, lexicon_path, lm_path):
    counts, vocab = arpa.read_counts_from_arpa(lm_path)
    symb = {v: k for k, v in vocab.items()}
    root = TrieNode('*')
    d = {}
    with open("lexicon.txt", "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        for l in lexicon:
            d[l[1:]] = l[0]
    for word in vocab:
        if word in d:
            add(root, d[word], word)

    g_lexicon = language_graph(tokens_to_index, root, vocab)

    g_lm = arpa.build_lm_graph(counts, vocab)
    g = gtn.compose(g_lexicon, g_lm)

