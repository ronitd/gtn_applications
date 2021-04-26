import gtn
import numpy as np
import math
#from IPython.display import display, Image
# import sys
# sys.path.append('./scripts/')
# import audioset
import scripts.load_arpa as arpa
import pywrapfst as fst
# from mfst import FST, RealSemiringWeight
import pickle
from collections import defaultdict

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
            
            graph.add_arc(inode, fst.Arc(ilabel=lbl, olabel=lbl, weight=fst.Weight(graph.weight_type(), math.exp(prob)), nextstate=onode))
            if bckoff is not None and vocab[EOS] not in ngram:
                bnode = get_node(ngram[1:])
                graph.add_arc(onode, fst.Arc(ilabel=epsilon, olabel=epsilon, weight=fst.Weight(graph.weight_type(), math.exp(bckoff)), nextstate=bnode))
    graph.arcsort(sort_type="ilabel")
    graph.write("G.fst")
    return graph


def build_lexicon_graph(tokens_to_idx, vocab, d, phoneme_to_word):
    epsilon = 0
    g = fst.VectorFst()
    s = g.add_state()
    g.set_start(s)
    f = g.add_state()
    g.set_final(f)
    print(g.arc_type())
    problem_idxs = []
    #problem_idxs=[806, 2009, 2030, 2376, 2442, 2831, 3470, 3557, 3803, 3899, 4414, 4481, 4695, 4715, 5424, 6310, 6558, 6611, 7073, 7683, 8134, 8387, 8805, 
    #      9010, 9218, 10194, 10619, 11208, 11882, 12019 ]
    for word_idx, word in enumerate(vocab):
        prev = s
        if word_idx in problem_idxs:
            print(word)
            print(d[word])
        #if word in d and word_idx<=len(vocab) and word_idx not in problem_idxs:
        if word in d and phoneme_to_word[tuple(d[word])][0]==word:
            for i, val in enumerate(d[word]):
                curr = g.add_state()
                oplabel = word_idx+1 if i == 0 else epsilon
                g.add_arc(prev, fst.Arc(ilabel=tokens_to_idx[val]+1, olabel=oplabel, nextstate=curr, weight=fst.Weight(g.weight_type(), 0)))
                prev = curr
            g.add_arc(prev, fst.Arc(ilabel=epsilon, olabel=epsilon, weight=fst.Weight(g.weight_type(), 0), nextstate=f))
    g.add_arc(f, fst.Arc(ilabel=epsilon, olabel=epsilon, weight=fst.Weight(g.weight_type(), 0), nextstate=s))
    print(g.num_states())

    #fst.disambiguate(g)
    #print("L Disambiguos")
    #print(g)
    #g = fst.determinize(g)
    #print("Determinize")
    g.arcsort(sort_type="olabel")
    g.write("L-Si.fst")
    

    text_file = open("L-Si.txt", "w")
    text_file.write(str(g))
    text_file.close()
    #exit()
    return g


def compose_language_grammar(tokens_to_index, lexicon_path, lm_path):
    counts, vocab = arpa.read_counts_from_arpa(lm_path)
    #print(vocab)
    #for i, val in enumerate(vocab):
    #    if i < 807:
    #        print(val)
    # symb = {v: k for k, v in vocab.items()}
    # root = TrieNode('*')
    d = {}
    phoneme_to_word = defaultdict(list)
    with open(lexicon_path, "r") as fid:
        lexicon = (l.strip().split() for l in fid)
        for l in lexicon:
            d[l[0]] = l[1:]
            phoneme_to_word[tuple(l[1:])].append(l[0])
    '''
    count = 0    
    for key, val in phoneme_to_word.items():
        if len(val)>1:
            count+=1
            print(val)
            print(key)
    print(count)
    exit()
    '''
    # for word in vocab:
    #     if word in d:
    #         add(root, d[word], word)
    #
    # g_lexicon = language_graph(tokens_to_index, root, vocab, len(tokens_to_index))
    
    g_lexicon= build_lexicon_graph(tokens_to_index, vocab, d, phoneme_to_word)
    print("Lexicon check: ", g_lexicon.verify())
    print("Arc type: ", g_lexicon.arc_type())
    print("Lexicon Done")
    
    g_lm = build_lm_graph(counts, vocab)
    print("LM check: ", g_lm.verify())
    print('G Done')

    
    LG = fst.compose(g_lexicon, g_lm)
    print("LG States: ", LG.num_states())
    #LG = fst.determinize(LG)
    #print("After Determinize states: ",LG.num_states() )
    #LG.minimize()
    #print("After Miniminize states: ",LG.num_states() )
    LG.arcsort(sort_type="ilabel")
    LG.write("LGDetMin.fst")
    text_file = open("gu-LG.txt", "w")
    text_file.write(str(LG))
    text_file.close()
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

