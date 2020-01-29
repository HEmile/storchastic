import storch

def walk_graph(node: storch.Tensor):
    counters = [1, 1, 1]
    names = {}
    def recur(node:storch.Tensor, counters, names):
        if node in names:
            return
        else:
            if node.stochastic:
                name = "s" + str(counters[0])
                counters[0] += 1
            elif node.is_cost:
                name = "c" + str(counters[1])
                counters[1] += 1
            else:
                name = "d" + str(counters[2])
                counters[2] += 1
            names[node] = name
        print(name, node)
        for p in node._parents:
            recur(p, counters, names)
            print(names[p] + "->" + name)

    recur(node, counters, names)