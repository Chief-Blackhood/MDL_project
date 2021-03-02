from client import get_errors, get_parents
import random
import json
import numpy as np


overfit_vector = [
    0.0,
    -1.45799022,
    -2.28980078,
    4.62010753e-1,
    -1.75214813e-2,
    -1.83669770e-15,
    8.52944060,
    2.29423303e-05,
    -2.04721003e-06,
    -1.59792834e-08,
    9.98214034e-10,
]
generated_vectors = []
probability = []
generations = 5
population = 20
mating_size = 20  # 10 - 15
parents_participating = 3  # 5 - 10


def get_initial_parents():
    top100 = get_parents(100)

    data = []

    for i in range(population):
        v = top100[random.randint(0, 99)]
        v = v["vector"]
        p1 = {"vector": v, "MSE": get_errors(v), "generation": 1}
        p1["score"] = get_score(p1)

        data.append(p1)

    return data


def main():
    # parents = get_initial_parents()
    parents = []
    with open("last_parent.txt", "r") as last_parent:
        for line in last_parent:
            parents.append(json.loads(line))

    print(parents)
    write_file = open("generations.txt", "a")

    for _ in range(generations):
        parent_file = open("last_parent.txt", "w")
        children = mate(parents)
        next_gen = get_next_gen(parents, children)
        for i in range(len(next_gen)):
            vector = []
            for v in next_gen[i]["vector"]:
                vector.append(v)
            next_gen[i]["vector"] = vector
            json.dump(next_gen[i], write_file)
            json.dump(next_gen[i], parent_file)
            parent_file.write("\n")
            write_file.write("\n")
        write_file.write("\n")


def get_score(data):
    MSE = data["MSE"]
    # a = (MSE[0]*0.1+0.9*MSE[1])
    # a = (MSE[0]*0.0+1.0*MSE[1])
    # a = MSE[0] * 0.2 + 0.8 * MSE[1]
    a = MSE[0] * 0.5 + 0.5 * MSE[1]
    # a = MSE[0] * 0.4 + 0.6 * MSE[1]
    # a = MSE[0] * 0.3 + 0.7 * MSE[1]
    # print(a)
    return a


def crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    deviation = 2  # 2 - 5
    chance = random.random()
    if chance < 0.5:
        beta = 2 * chance ** (1 / (deviation + 1))
    else:
        beta = (1 / (2 * (1 - chance))) ** (1 / deviation + 1)

    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    return child1, child2


def mate(all_parents):
    scores = [p["score"] for p in all_parents]
    scores = np.array(scores)
    all_parents = np.array(all_parents)
    sorted_parents = all_parents[(np.argsort(scores))]
    good_parents = sorted_parents[:mating_size]
    generation = all_parents[0]["generation"]

    children = []

    for _ in range(population // 2):
        parent1 = good_parents[random.randint(0, mating_size - 1)]
        parent2 = good_parents[random.randint(0, mating_size - 1)]

        child1, child2 = crossover(parent1["vector"], parent2["vector"])

        child1 = mutate(child1)
        child2 = mutate(child2)

        child1 = {"generation": generation, "MSE": get_errors(child1), "vector": child1}
        child2 = {"generation": generation, "MSE": get_errors(child2), "vector": child2}

        child1["score"] = get_score(child1)
        child2["score"] = get_score(child2)

        children.append(child1)
        children.append(child2)

    return children


def mutate(child):
    for i in range(11):
        chance = random.random()
        if chance < 0.3:
            scale = 1
            if i in [7, 9]:
                scale = 1 * 1e-3  # for less change 1e-3
            elif i in [8, 10]:
                scale = 1 * 1e-4  # for less change 1e-4
            elif i in [5]:
                scale = 1e-2  # for less change 1e-2
            scale *= 5  # for large mutation
            random_add = (random.random() - 0.5) * scale
            if i == 0:
                child[i] = random_add
            else:
                child[i] = child[i] + child[i] * random_add

    return child


def get_next_gen(parents, children):
    children_scores = [p["score"] for p in children]
    parents_scores = [p["score"] for p in parents]

    children = np.array(children)
    parents = np.array(parents)

    children = children[np.argsort(children_scores)]
    parents = parents[np.argsort(parents_scores)]

    selected_children = children[: (population - parents_participating)]
    selected_parents = parents[:parents_participating]

    next_gen = np.concatenate((selected_children, selected_parents))

    for i in range(len(next_gen)):
        next_gen[i]["generation"] = next_gen[i]["generation"] + 1

    return next_gen


main()
# print(get_parents(1))
