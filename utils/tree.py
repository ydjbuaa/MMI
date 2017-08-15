# tree object from stanfordnlp/treelstm
from utils.vocab import DepTransitions


class Tree(object):
    def __init__(self):
        self.parent = None
        self.idx = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def child_enriched_structure(self):
        ret = [self.idx]
        if self.num_children > 0:
            for i in range(self.num_children):
                ret += self.children[i].child_enriched_structure()
        return ret

    def head_enriched_structure(self):
        ret = []
        if self.num_children > 0:
            for i in range(self.num_children):
                ret += self.children[i].head_enriched_structure()
        ret += [self.idx]
        return ret


def read_tree(tree_words):
    parents = list(map(int, tree_words))
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        # if not trees[i-1] and parents[i-1]!=-1:
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                # if trees[parent-1] is not None:
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root


def get_corresponding_order(order):
    order_dict = {}
    for i, idx in enumerate(order):
        order_dict[idx] = i
    return [order_dict[i] for i in range(len(order))]
    pass


def tree2transition(tree_words):
    parents = list(map(int, tree_words))
    stack = [-1]
    relations = {}
    child_num = {-1: 0}
    for i in range(len(tree_words)):
        child_num[i] = 0

    for i, pid in enumerate(parents):
        pid = pid - 1
        child_num[pid] += 1
        relations[str(i)+'\t'+str(pid)] = DepTransitions.RR
        relations[str(pid)+'\t'+str(i)] = DepTransitions.LR

    trans = []
    for i in range(len(parents)):
        stack.append(i)
        trans.append(DepTransitions.SH)
        while len(stack) >= 2:
            key = str(stack[-1])+'\t'+str(stack[-2])
            if key in relations:
                if relations[key] == DepTransitions.LR and child_num[stack[-2]] > 0:
                    break
                if relations[key] == DepTransitions.RR and child_num[stack[-1]] > 0:
                    break
                trans.append(relations[key])
                w1 = stack.pop()
                w2 = stack.pop()

                if trans[-1] == DepTransitions.LR:
                    stack.append(w1)
                    child_num[w1] -= 1
                else:
                    stack.append(w2)
                    child_num[w2] -= 1
            else:
                break

    assert len(trans) == 2 * len(tree_words)
    return trans


if __name__ == '__main__':
    tree_words = "4 4 4 0 6 4 4 4 10 8 10 8".split()
    print(' '.join(map(str, tree2transition(tree_words))))
    pass
