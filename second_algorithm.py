from client_validation import get_errors, get_parents
from client_test import submit
import random
import json
import numpy as np

first_gen = False


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
rank20_vector = [
    -0.5315347166952884,
    -2.0522509655713694,
    -3.1692939908232605,
    6.1450778806927575,
    -0.050785981221236225,
    -1.8218574931408493e-15,
    -0.1756376755478551,
    2.9528452684408868e-05,
    -2.117280506724165e-06,
    -1.701803847854941e-08,
    9.021927460810928e-10,
]
goodrank25_vector = [
    -9.09008485495793,
    4.842398546864905,
    -1.6593932871285404,
    6.03983343064555,
    -1.0371339627715168,
    -1.8469794957039708e-15,
    0.37312278547894107,
    2.9184881000925245e-05,
    -2.0638474169621287e-06,
    -1.526937696656356e-08,
    9.190051080273302e-10,
]
generated_vectors = []
probability = []
generations = 1
population = 20
mating_size = 10  # 10 - 15
parents_participating = 5  # 5 - 10


def out_to_file(text, vectors=None):
    file = open("submission.txt", "a")

    file.write(text)
    file.write("\n")

    if not vectors:
        # file.write(str([1, 2, 3]))
        return

    for vec in vectors:
        vector = []
        for v in vec:
            vector.append(v)
        # json.dump({"vector": vector}, file)
        file.write(str(vector))
        file.write("\n")

    file.write("\n")


def get_initial_parents():
    # top100 = get_parents(100)

    data = []

    for i in range(population):
        # v = top100[random.randint(0, 99)]
        # p1["score"] = get_score(p1)
        # v = v["vector"]
        # p1 = {"vector": v, "MSE": get_errors(v), "generation": 1}
        # p1["score"] = get_score(p1)
        copy = []
        for j in range(len(goodrank25_vector)):
            copy.append(goodrank25_vector[j])

        v = mutate(copy)
        p1 = {"generation": 1, "MSE": get_errors(v), "vector": v}
        p1["score"] = get_score(p1)
        data.append(p1)

    return data


def main():
    if first_gen:
        parents = get_initial_parents()
    else:
        parents = []
        with open("last_parent_leaderboard.txt", "r") as last_parent:
            for line in last_parent:
                parent = json.loads(line)
                parent["score"] = get_score(parent)
                parents.append(parent)

    # print(parents)
    write_file = open("generations_leaderboard.txt", "a")

    for _ in range(generations):
        out_to_file("--------------------")
        out_to_file("Generation: " + str(parents[0]["generation"]))
        out_to_file("")
        parent_file = open("last_parent_leaderboard.txt", "w")
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
    a = MSE[0] + MSE[1]
    # a = (MSE[0]*0.1+0.9*MSE[1])
    # a = (MSE[0]*0.0+1.0*MSE[1])
    # a = MSE[0] * 0.2 + 0.8 * MSE[1]
    # a = MSE[0] * 0.5 + 0.5 * MSE[1]
    # a = MSE[0] * 0.4 + 0.6 * MSE[1]
    # a = MSE[0] * 0.3 + 0.7 * MSE[1]
    train_min = 38e11
    train_max = 8e12
    val_min = 38e11
    val_max = 8e12

    for i in range(15, -1, -1):
        factor = 1.2
        if (
            MSE[0] < train_min / (factor ** i)
            or MSE[0] > train_max * (factor ** i)
            or MSE[1] < val_min / (factor ** i)
            or MSE[1] > val_max * (factor ** i)
        ):
            return (i + 1) * 1e15

    # if MSE[0] < 34e11 or MSE[0] > 12e12 or MSE[1] < 14e11 or MSE[1] > 10e12:
    #     return 8e15

    # if MSE[0] < 36e11 or MSE[0] > 10e12 or MSE[1] < 16e11 or MSE[1] > 8e12:
    #     return 7e15

    # if MSE[0] < 38e11 or MSE[0] > 8e12 or MSE[1] < 18e11 or MSE[1] > 6e12:
    #     return 6e15

    # if MSE[0] < 40e11 or MSE[0] > 6e12 or MSE[1] < 20e11 or MSE[1] > 4e12:
    #     return 5e15

    # if a > 1e14 or a < 1e10:
    #     return 0
    # if a > 1e13 or a < 1e11:
    #     return -1
    # if a > 35e12 or a < 35e10:
    #     return -2
    # print(a)
    return a


def crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    deviation = 4  # 2 - 5
    chance = random.random()
    if chance < 0.5:
        beta = 2 * chance ** (1 / (deviation + 1))
    else:
        beta = (1 / (2 * (1 - chance))) ** (1 / deviation + 1)

    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    for i in range(len(child1)):
        if abs(child1[i]) > 10 or abs(child2[i]) > 10:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return child1, child2


def mate(all_parents):
    scores = [p["score"] for p in all_parents]
    parent_vectors = [p["vector"] for p in all_parents]
    out_to_file("Initial Population", parent_vectors)

    scores = np.array(scores)
    all_parents = np.array(all_parents)
    sorted_parents = all_parents[(np.argsort(scores))]
    good_parents = sorted_parents[:mating_size]
    generation = all_parents[0]["generation"]

    children = []
    submission_crossover = []
    submission_selection = []
    submission_mutation = []

    for _ in range(population // 2):
        parent1 = good_parents[random.randint(0, mating_size - 1)]
        parent2 = good_parents[random.randint(0, mating_size - 1)]

        submission_selection.append(parent1["vector"])
        submission_selection.append(parent2["vector"])

        child1, child2 = crossover(parent1["vector"], parent2["vector"])

        submission_crossover.append(child1)
        submission_crossover.append(child2)

        child1 = mutate(child1)
        child2 = mutate(child2)

        submission_mutation.append(child1)
        submission_mutation.append(child2)

        child1 = {"generation": generation, "MSE": get_errors(child1), "vector": child1}
        child2 = {"generation": generation, "MSE": get_errors(child2), "vector": child2}

        child1["score"] = get_score(child1)
        child2["score"] = get_score(child2)

        children.append(child1)
        children.append(child2)

    out_to_file("After Selection", submission_selection)
    out_to_file("After Crossover", submission_crossover)
    out_to_file("After Mutation", submission_mutation)

    return children


def mutate(child, prob=0.3):
    for i in range(11):
        chance = random.random()
        if chance < prob:
            scale = 1
            if i in [7, 9]:
                scale = 10 * 1e-3  # for less change 1e-3
            elif i in [8, 10]:
                scale = 10 * 1e-4  # for less change 1e-4
            elif i in [5]:
                scale = 1e-2  # for less change 1e-2
            # scale *= 10  # for large mutation
            random_add = (random.random() - 0.5) * scale
            if i == 0:
                if abs(random_add) <= 10:
                    child[i] = random_add
            else:
                if abs(child[i] + child[i] * random_add) <= 10:
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


# main()

# print(get_parents(10))
# lol = get_parents(20)
# for l in lol:
#     print(l)
#     print()
# print(get_errors([
#             -0.5201868916958766,
#             -1.188706080024647,
#             -2.8811179052944116,
#             3.068755758956475,
#             -0.02688982912822353,
#             -1.856825758088662e-15,
#             -0.031191811182051266,
#             2.2624328388553044e-05,
#             -2.03280561689875e-06,
#             -1.6106880131916125e-08,
#             9.311835351664468e-10,
#         ]))
# leaderboard
print(
    submit(
        [2.096435824649727, 6.854410010154593, -3.047649288276091, 6.039833430645549, -1.0772600799101457, -1.825221801210638e-15, 1.2017084638551592, 2.9526992539482523e-05, -2.069692770325736e-06, -1.5029303133528182e-08, 9.226294164167137e-10]
    )
)
# rank 25 error 4.7e12
# {"generation": 12, "MSE": [4635577827633.479, 4962072117899.779], "vector": [2.096435824649727, 6.854410010154593, -3.047649288276091, 6.039833430645549, -1.0772600799101457, -1.825221801210638e-15, 1.2017084638551592, 2.9526992539482523e-05, -2.069692770325736e-06, -1.5029303133528182e-08, 9.226294164167137e-10], "score": 9597649945533.258}

# temp = []
# for _ in range(11):
#     temp.append(random.random() * 20 - 10)
# print(submit(temp))


# rank 73, error 3e13
# print(submit([2.285327019919199, 9.838006123873061, 0.8709570011926488, 7.262351942722824, 0.43730521498916797, 2.8292824313203438e-15,
#   1.0057376885881602, 7.014686372405952e-05, -3.711585930580224e-06, -3.8504293737417424e-08, 1.6197655803475092e-09]))


# rank 43, error 1.5e13
# print(submit([1.0432186791899198, 1.6594465241711451, -4.969513834628463, 6.650790547146654, 0.18929331692924523, 2.0292427762543304e-15,
#   0.07717475565215004, 5.490705016161995e-05, -3.615580917503483e-06, -2.915977080647393e-08, 1.5564728356587447e-09]))


# rank 55, error 2.1e13
# print(submit([-1.7618721292125337, 1.5254518673454647, 3.8056243294170375, 9.33956619012714, 0.19539868188257653, 2.3521692885851494e-15,
#   0.7254059637825796, 6.213129542739528e-05, -3.6542358823781994e-06, -3.293149683495686e-08, 1.6038933256367805e-09]))


# rank 48, error 2e13
# print(submit([-0.3205872238700628, 4.722200089572409, 2.5827252608325324, 0.020742580513831474, 0.025097341631950192, 1.9497600858059046e-15,
#   0.7651725742995477, 5.863728180731816e-05, -3.691865749333086e-06, -3.149157107254597e-08, 1.6097730454748171e-09]))


# rank 44, error 1.8e13
# print(submit([0.3224901477365446, 5.863411530108603, 1.915364487609244, 2.1295836181608743, 0.05795762382107966, 1.959919641686096e-15,
#   0.49794634447394515, 5.695692875098175e-05, -3.627369668657371e-06, -3.027775874665646e-08, 1.5774165723081916e-09]))


# rank 41, error 1.1e13
# print(submit([0.18509960216156252, 5.766666134721964, 2.7857835051379194, 6.348583455866942, 0.10996769893349412, 1.8252345423220324e-15,
#   1.143656426680526, 6.31684571083148e-05, -3.62736966865736e-06, -3.027775874665644e-08, 1.5413717034164426e-09]))


# rank 43, error 1.7e13
# print(submit([0.3462842289483563, 5.742079094036775, 2.210963591139381, 3.035967991455211, 0.08565256591374334, 2.0384615740615944e-15,
#   0.7430886539498689, 5.778893077179256e-05, -3.6299341157036637e-06, -3.07079468416999e-08, 1.572570043581758e-09]))


# rank 48, error 1.8e13
# print(submit([0.29538683361033263, 5.58779061362412, 1.645637283193234, 1.464191407633081, 0.07622008705618795, 1.9384005601751386e-15,
#   0.4764361685384172, 5.675031585467867e-05, -3.62736966865737e-06, -3.027775874665647e-08, 1.5786155259843508e-09]))


# print(
#     submit(
#         [
#             1.0131872505490749,
#             -2.462818793194241,
#             0.15717107252282003,
#             9.305804527364357,
#             -0.004032359329763721,
#             -1.8266231882472374e-15,
#             0.0803572819261498,
#             2.9523627015813384e-05,
#             -2.1173454173673293e-06,
#             -1.7041966866366392e-08,
#             9.022976062882336e-10,
#         ]
#     )
# )
# rank 39 error: 11058159841510
# [
#             -0.27731032659947613,
#             0.5644836520084715,
#             -9.285899062831763,
#             9.915976191511891,
#             -0.014335631159441085,
#             -2.234327316779149e-15,
#             0.04722108861486259,
#             9.008444555589758e-06,
#             -1.9379919103286594e-06,
#             -8.794224685634096e-09,
#             8.415787410247279e-10,
#         ]
# rank 37 error 8e12
# {"generation": 42, "MSE": [3473725761136.134, 4048044419582.585], "vector": [-0.5315347166952884, -2.0522509655713694, -3.1692939908232605, 6.1450778806927575, -0.050785981221236225, -1.8218574931408493e-15, -0.1756376755478551, 2.9528452684408868e-05, -2.117280506724165e-06, -1.701803847854941e-08, 9.021927460810928e-10], "score": 1000000000000000.0}
# rank 20 error 4e12
# {"generation": 61, "MSE": [642968584373.1306, 2926422215987.839], "vector": [-0.06723646215001355, -1.0329749111105289, -7.550413484104738, 6.211556994478153, -0.05538165193189995, -3.1908079940589373e-15, -0.045310849061531155, 6.683923992143137e-06, -1.5700378768886358e-06, -2.507984710652146e-09, 7.183167446734617e-10], "score": 7000000000000000.0}
# rank 8 error 1e12
# {"generation": 68, "MSE": [6908776934740.484, 22545548555289.273], "vector": [-0.9547480995659099, 6.978886637320509, -4.2845236664935555, 6.671563601294054, -0.16262919159115857, -5.799461942157361e-15, -0.7461172030332495, -7.00604854180074e-06, -4.860430714625869e-07, 1.0946860966190178e-08, 2.451990037147383e-10], "score": 3000000000000000.0}
