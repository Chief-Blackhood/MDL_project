from client import get_errors, get_parents
import random
import json


# [13510723202.57021, 368296581138.17303]1e10, 1e11

# highly sensitive = 8 (1e-4),10(1e-4)
# medium-high = 7(1e-3),9(1e-3)
# medium = 5 (1e-2)
# medium-low = 3 (1e10), 4(1e8)
# low = 1 (1e12),2 (1e13)
# least = 6 (1e16), 0([-10, 10])

overfit_vector = [
    0.0,
    -1.45799022e-12,
    -2.28980078e-13,
    4.62010753e-11,
    -1.75214813e-10,
    -1.83669770e-15,
    8.52944060e-16,
    2.29423303e-05,
    -2.04721003e-06,
    -1.59792834e-08,
    9.98214034e-10,
]
gfather_vector = [
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

# [0.0, -10, -2.717605255632141, -10, -0.013537570582544144, 5.212509755825319e-16, 10, 3.3512490168006345e-08,
#  -4.5882491141982304e-09, -2.0850649600366798e-11, 6.56545780363502e-14]

generated_vectors = []
probability = []
population = 16


def main():
    parent_data = get_parents(population)
    # print(parent_data)
    with open("generation.txt", "a") as write_parent:
        for line in parent_data:
            json.dump(line, write_parent)
            write_parent.write("\n")
        write_parent.write("\n")

    probability_of_choosing(parent_data)
    children = get_children(parent_data)
    open("parent.txt", "w").close()
    write_file = open("parent.txt", "a")
    for i in range(population):
        MSE = get_errors(children[i])
        json_data = {
            "MSE": MSE,
            "score": 1 / get_score(MSE),
            "generated_vector_used": children[i],
        }
        json.dump(json_data, write_file)
        if i != population - 1:
            write_file.write("\n")


def get_score(MSE):
    # a = (MSE[0]*0.1+0.9*MSE[1])
    # a = (MSE[0]*0.0+1.0*MSE[1])
    # a = MSE[0] * 0.2 + 0.8 * MSE[1]
    a = MSE[0] * 0.5 + 0.5 * MSE[1]
    # a = MSE[0] * 0.4 + 0.6 * MSE[1]
    # a = (MSE[0]*0.3+0.7*MSE[1])
    # print(a)
    return 1 / a


def probability_of_choosing(parent_data):
    total_score = 0
    for i in range(population):
        total_score += get_score(parent_data[i]["MSE"])
    for i in range(population):
        probability.append(get_score(parent_data[i]["MSE"]) / total_score)


def get_parents_index(parent_data):
    selected = [-1, -1]

    while selected[0] == -1 or selected[1] == -1:
        to_check = random.randint(0, population - 1)
        chance = random.random()

        if chance < probability[to_check]:
            if selected[0] == -1:
                selected[0] = to_check
            elif to_check != selected[0]:
                selected[1] = to_check

    return selected


def crossover_children(selected_parent_data):
    for i in range(population):
        if i in [4, 5, 7, 8]:
            temp = selected_parent_data[0][i]
            selected_parent_data[0][i] = selected_parent_data[1][i]
            selected_parent_data[1][i] = temp
    return selected_parent_data


def mutate_children(children_data):
    for i in range(population):
        for j in range(11):
            chance = random.random()
            if chance < 0.7:
                # print('mutate')
                scale = 1
                if i in [7, 9]:
                    scale = 1 * 1e-2  # for less change 1e-3
                elif i in [8, 10]:
                    scale = 1 * 1e-2  # for less change 1e-4
                elif i in [5]:
                    scale = 1e-2  # for less change 1e-2
                if random.random() < 0.1:
                    scale *= 2
                if random.random() < 0.1:
                    scale *= 2
                random_add = (random.random() - 0.5) * scale
                if random.random() < 0.2:
                    scale *= 7
                if j == 0:
                    children_data[i][j] = random_add
                else:
                    children_data[i][j] = (
                        children_data[i][j] + children_data[i][j] * random_add
                    )
    return children_data


def get_children(parent_data):
    children_data = []
    for _ in range(population // 2):
        selected = get_parents_index(parent_data)
        parent_1 = parent_data[selected[0]]["generated_vector_used"]
        parent_2 = parent_data[selected[1]]["generated_vector_used"]
        children = crossover_children([parent_1, parent_2])
        children_data.append(children[0])
        children_data.append(children[1])
    children_data = mutate_children(children_data)
    return children_data


################################
## function calls for genetic algorithm
################################

main()

################################
## Debugging
################################


def generate_parents():
    for _ in range(population):
        generated_vector = []

        for i in range(len(gfather_vector)):
            scale = 1
            if i in [7, 9]:
                scale = 1e-2  # for less change 1e-3
            elif i in [8, 10]:
                scale = 1e-2  # for less change 1e-4
            elif i in [5]:
                scale = 1e-1  # for less change 1e-2
            random_add = (random.random() - 0.5) * scale
            new_value = gfather_vector[i] + gfather_vector[i] * random_add
            generated_vector.append(max(min(new_value, 10), -10))

        generated_vectors.append(generated_vector)

    write_file = open("parent.txt", "a")
    for i in range(population):
        MSE = get_errors(generated_vectors[i])
        print(MSE)
        json_data = {
            "MSE": MSE,
            "score": 1 / get_score(MSE),
            "generated_vector_used": generated_vectors[i],
        }
        json.dump(json_data, write_file)
        write_file.write("\n")


def gfather_test():
    write_file = open("gfather.txt", "a")
    MSE = get_errors(gfather_vector)
    json_data = {
        "MSE": MSE,
        "score": 1 / get_score(MSE),
        "generated_vector_used": gfather_vector,
    }
    json.dump(json_data, write_file)
    write_file.write("\n")


def variation_test(index):
    # mutation_pos = random.random() * 0.05
    # mutation_neg = - random.random() * 0.05
    mutation_pos = 10 ** -4
    write_file = open("variation.txt", "a")

    overfit = []
    for d in overfit_vector:
        overfit.append(d)

    overfit[index] += overfit[index] * mutation_pos
    MSE = get_errors(overfit)
    json_data = {
        "mutation": "+" + str(index),
        "value": mutation_pos,
        "MSE": MSE,
        "score": 1 / get_score(MSE),
        "generated_vector_used": overfit,
    }
    json.dump(json_data, write_file)
    write_file.write("\n")

    # overfit = []
    # for d in overfit_vector:
    #     overfit.append(d)

    # overfit[index] += overfit[index] * mutation_neg
    # MSE = get_errors(overfit)
    # json_data = {
    #     'mutation': '-' + str(index),
    #     'value': mutation_neg,
    #     'MSE': MSE,
    #     'score': 1 / get_score(MSE),
    #     'generated_vector_used': overfit
    # }
    # json.dump(json_data, write_file)
    # write_file.write('\n')


def variation_test_for_zero():
    write_file = open("variation.txt", "a")

    overfit = []
    for d in overfit_vector:
        overfit.append(d)

    mutation_pos = -10
    overfit[0] += mutation_pos
    MSE = get_errors(overfit)
    json_data = {
        "mutation": "+" + "0",
        "value": mutation_pos,
        "MSE": MSE,
        "score": 1 / get_score(MSE),
        "generated_vector_used": overfit,
    }
    json.dump(json_data, write_file)
    write_file.write("\n")


################################
## function calls for debugging
################################

# for i in range(1, 11):
#     variation_test(i)

# variation_test(1)
# variation_test(3)
# variation_test(4)
# variation_test(5)
# variation_test(7)
# variation_test(8)
# gfather_test()
# variation_test(9)
# variation_test(10)
# variation_test(2)
# variation_test(6)
# variation_test_for_zero()

# generate_parents()

# lol = get_errors((get_parents(1))[0])
# print(lol)
# print(1/get_score(lol))
