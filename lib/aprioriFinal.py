import pandas as pd
import numpy as np
import math
import operator
import itertools
import collections
from itertools import groupby
from collections import Counter
from itertools import combinations, chain

class ItemSet:
    items = []
    def __init__(self, items, count):
        self.items = items
        if(count is None):
            self.count = 0
        else:
            self.count = count

    def __str__(self):
        return str(self.items) + " count " + str(self.count)
    
    def __getCount__(self):
        return self.count
    
    def __getItems__(self):
        return self.items
    
    def __setCount__(self, count):
        self.count = count

# -- Only used for smaller tests -- #
def load_dataset():
    return [['d3javascript', 'alpha', 'beta', 'c++', 'e'],
            ['alpha', 'c++', 'e'],
            ['beta', 'c++', 'e'],
            ['alpha', 'e'],
            ['alpha', 'c++', 'd3javascript'],
            ['beta', 'c++', 'e'],
            ['beta', 'c++', 'e'],
            ['c++', 'd3javascript', 'e'],
            ['d3javascript', 'e'],
            ['beta'],
            ['beta', 'c'],
            ['b', 'c', 'd3javascript'],
            ['c', 'd', 'e']]

def find_frequent_1_itemsets(D, threshold):
    C1 = {}
    for itemset in D:
        for item in itemset:
            # print item, type(item),  item in C1, item, C1, itemset
            # -- Count support for each item -- #
            if item in C1:
                C1[item] += 1
            else:
                C1[item] = 1
    # -- Return the items that have support bigger than the threshold -- #
    return {hash(ItemSet([item], C1[item])): ItemSet([item], C1[item]) for item in C1 if C1[item] >= threshold}

def joinable(itemset1, itemset2):
    # -- Join the two itemsets if they differ only by the last symbol -- #
    diff = set(itemset2.__getItems__()) - set(itemset1.__getItems__())
    if len(diff) is not 1:
        return None
    return ItemSet(sorted(itemset1.__getItems__()) + list(diff), 0)

def has_infrequent_subset(c, L): # c: candidate k-itemset; L: frequent (k-1) itemsets); // use prior knowledge
    k_min_one = len(c) - 1
    L_itemsets = []
    for itemSet in L:
        L_itemsets.append(itemSet.items)
    for subSet in [list(t) for t in itertools.combinations(c, k_min_one)]:
        # -- Check if there exists any infrequent subset -- #
        if not subSet in L_itemsets:
            return True
    return False

def apriori_gen(L): # L of (k-1)
    Ck = {}
    for itemset_l1 in sorted(L.itervalues()):
        for itemset_l2 in sorted(L.itervalues()):
            c = joinable(itemset_l1, itemset_l2) # join the itemsets

            if c and not has_infrequent_subset(c.items, L.itervalues()): # check if c.items contain any infrequency in the subsets
                c = ItemSet(sorted(c.items), 0)
                Ck[tuple(c.items)] = c  # init candidate and check in main() how frequently it appears in the transactions set
    return Ck

def apriori(data_set, min_sup):
    # -- Init all levels' dictionary -- #
    L = {}
    # -- Find level 1 of frequent itemsets -- #
    L[1] = find_frequent_1_itemsets(data_set, min_sup)  # scan D for count of each candidate
    k = 2
    # -- Continue with upper levels until the levels where no more candidates are created -- #
    while len(L[k - 1]) > 0:
        Ck = apriori_gen(L[k - 1])
        for t in data_set:
            for c in Ck.itervalues():
                if set(c.items).issubset(set(t)):
                    Ck[tuple(c.items)].__setCount__(Ck[tuple(c.items)].__getCount__() + 1)
            L[k] = {tuple(item.items): item for item in Ck.itervalues() if item.__getCount__() >= min_sup}
        k += 1
    return L


def get_result(L):
    final_L = []
    result_dict = {}
    # -- Iterate through all the levels and add the itemsets to a list for printing -- #
    for level in L.itervalues():
        for itemSet in level.itervalues():
            if (itemSet.items, itemSet.count) not in final_L:
                final_L.append((itemSet.items, itemSet.count))
    # -- Hash all itemsets to their support value, so it is easy to generate asociation rules later -- #
    for result_instance in final_L:
        result_dict[tuple(result_instance[0])] = result_instance[1]

    return result_dict

def find_subsets(S,m):
    return set(itertools.combinations(S, m))

def get_association_rules_for_frequent_itemset(itemset_count_tuple, all_frequent_itemsets, percentage_min_limit):
    frequent_itemset = sorted(set(itemset_count_tuple))

    # -- Generate all nonempty subsets of frequent itemset, book page 254 -- #
    all_nonempty_subsets = list()
    for k in range(1, len(frequent_itemset)):
        level_subsets = find_subsets(frequent_itemset, k)
        for subset in level_subsets:
            all_nonempty_subsets.append(sorted(subset))
    # -- Output association rule if it satisfies a min_conf (book, page 254) -- #
    for subset in all_nonempty_subsets:
        subset_vs_freq_itemset__symmetric_diff = sorted(set(subset) ^ set(frequent_itemset))
        percentage = float(all_frequent_itemsets[tuple(itemset_count_tuple)]) / float(all_frequent_itemsets[tuple(subset_vs_freq_itemset__symmetric_diff)])
        association = str(subset_vs_freq_itemset__symmetric_diff) + " => " + str(subset)
        if percentage >= percentage_min_limit:
            print association + ", ", str(percentage * 100.0) + "%"


def main(D, min_sup, percentage_min_limit):
    # -- Run the algorithm && get all levels of candidates -- #
    L = apriori(D, min_sup)
    # -- Get the result -- #
    result_dict = get_result(L)

    # -- Print association rules -- #
    for r in result_dict.iterkeys():
        get_association_rules_for_frequent_itemset(list(r), result_dict, percentage_min_limit)

# D = load_dataset()
# min_sup = 2  # must change it with trial-and-error
# main(D, min_sup, 0.8)

