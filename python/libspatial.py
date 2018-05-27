import numpy as np
from keras.models import Sequential
from keras.layers import Dense

ca_min_radius = 2
ca_max_radius = 2


def ca_get_radius(lut):
    return int((np.log2(lut.shape[0])-1)/2)


def ca_lut_len(radius):
    return 2**(2*radius+1)


def ca_decode_rule(rule_num, radius):
    return np.array([int(c) for c in bin(rule_num)[2:].zfill(ca_lut_len(radius))]).astype(np.int8)


def ca_decode_rule_bool(rule_num, radius):
    return [bool(c) for c in bin(rule_num)[2:].zfill(ca_lut_len(radius))]


def tool_random_binary_vector(lenght):
    return np.random.random(lenght).round().astype(np.int8)


def ca_init_random(radius):
    return tool_random_binary_vector(ca_lut_len(radius))


def ca_downscale_lut(lut, check=True):
    if(check):
        min_radius = ca_min_radius
    else:
        min_radius = 1

    radiusBig = ca_get_radius(lut)

    if(radiusBig <= min_radius):
        return lut

    lenBig = lut.shape[0]
    lenSmall = int(lenBig / 4)
    black_count = np.zeros(lenSmall, dtype=np.int8)

    for i in range(0, lenBig):
        if(lut[i] != 0):
            black_count[(i/2) % lenSmall] += 1

    for i in range(0, lenSmall):
        if(black_count[i] < 2):
            black_count[i] = 0
        elif(black_count[i] > 2):
            black_count[i] = 1
        else:
            black_count[i] = np.random.randint(0, 2)

    return black_count


def ca_upscale_lut(lut, check=True):
    radiusSmall = ca_get_radius(lut)

    if(check == True and radiusSmall >= ca_max_radius):
        return lut

    lenSmall = lut.shape[0]
    lenBig = lenSmall * 4
    lenBigHalf = int(lenBig / 2)
    lutBig = np.zeros(lenBig, dtype=np.int8)
    for i in range(0, lenSmall):
        a = lut[i]
        lutBig[2*i] = a
        lutBig[2*i+1] = a
        lutBig[lenBigHalf + 2*i] = a
        lutBig[lenBigHalf + 2*i+1] = a

    return lutBig


def ca_set_radius(lut, radius):
    if(radius < ca_min_radius):
        radius = ca_min_radius

    if(radius > ca_max_radius):
        radius = ca_max_radius

    current_radius = ca_get_radius(lut)
    diff = radius - current_radius

    if(diff > 0):
        for _ in range(0, diff):
            lut = ca_upscale_lut(lut)
    else:
        for _ in range(0, -1*diff):
            lut = ca_downscale_lut(lut)

    return lut


def ca_dist(lut1, lut2, upscale=False):
    l1 = lut1
    l2 = lut2
    if(l1.shape[0] != l2.shape[0]):
        if(not upscale):
            return -1
        else:
            r1 = ca_get_radius(lut1)
            r2 = ca_get_radius(lut2)
            radius = max(r1, r2)
            l1 = ca_set_radius(lut1, radius)
            l2 = ca_set_radius(lut2, radius)

    return np.abs(l1 - l2).sum()


def ca_equal(lut1, lut2):
    return ca_dist(lut1, lut2) == 0


def ca_get_pows(len):
    t = 2*len+1
    return np.array([2**x for x in range(t-1, -1, -1)])


def ca_conf_nei(conf, index, r):
    if(index - r >= 0 and index+r+1 < conf.shape[0]):
        nei = conf[index - r: index + r + 1]
    elif(index - r < 0):
        nei = np.concatenate((conf[index - r:], conf[0:index+r+1]))
    else:
        nei = np.concatenate((conf[index-r:], conf[0:index+r+1-conf.shape[0]]))
    return nei


def ca_apply(lut, input):
    radius = ca_get_radius(lut)
    pows = global_pows[radius-ca_min_radius]
    input_len = input.shape[0]

    idxs = (pows * np.array([ca_conf_nei(input, i, radius)
                             for i in range(0, input_len)])).sum(1)

    return np.take(lut, idxs)


def conf_diff(a, b):
    idxs = np.where(np.logical_and(a >= 0, b >= 0))[0]
    return np.abs(np.take(a, idxs) - np.take(b, idxs)).sum()


def conf_is_complete(conf):
    return conf.min() >= 0

# TODO: this needs refactoring, eliminating for loop should be possible


def conf_complete(src, comp):
    result = np.copy(src)
    l = result.shape[0]
    for t in range(0, l):
        if(result[t] < 0):
            result[t] = comp[t]
    return result


def id_error_step(lut, input, target_output):
    output = ca_apply(lut, input)
    return (conf_diff(target_output, output), output)


def id_error_step_gap(lut, input, target_output, max_gap=10):
    error = np.zeros(max_gap+1, dtype=np.int64)
    current_input = input
    outputs = []

    for i in range(0, max_gap+1):
        e, current_input = id_error_step(lut, current_input, target_output)
        error[i] = e
        outputs.append(current_input)

    min_err = error.min()
    if(conf_is_complete(target_output)):
        return (min_err, outputs[error.argmin()])

    return (min_err, outputs[np.random.choice(np.where(error == min_err)[0])])


def id_error(lut, image, max_gap=10):
    input = image[0]
    time_steps = image.shape[0]
    result = 0
    for t in range(0, time_steps-1):
        output = image[t+1]
        err, rule_output = id_error_step_gap(lut, input, output, max_gap)
        result += err
        if(not conf_is_complete(output)):
            input = conf_complete(output, rule_output)
        else:
            input = output

    return result

# TODO: this needs refactoring to eliminate for loop


def evolve_mutate(lut, pm):
    if(ca_min_radius < ca_max_radius and np.random.rand() < pm):
        ca_upscale_lut(lut)

    l = lut.shape[0]

    for t in range(0, l):
        if(np.random.rand() < pm):
            lut[t] = 1 - lut[t]

    if(ca_min_radius < ca_max_radius and np.random.rand() < pm):
        ca_downscale_lut(lut)

    return lut


def evolve_cross(lut1, lut2):
    if(lut1.shape[0] == lut2.shape[0]):
        p = np.random.random(lut1.shape[0]).round().astype(np.int8)
        return p*lut1 + (1-p)*lut2
    else:
        r1 = ca_get_radius(lut1)
        r2 = ca_get_radius(lut2)
        target_r = np.random.randint(min(r1, r2), max(r1, r2)+1)
        return evolve_cross(ca_set_radius(lut1, target_r), ca_set_radius(lut2, target_r))


def neural_get_model(afun='relu', layer_count=1):
    input_len = ca_lut_len(ca_max_radius)
    model = Sequential()
    model.add(Dense(input_len, input_dim=input_len, activation=afun))

    for _ in range(0, layer_count):
        model.add(Dense(256, activation=afun))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def neural_train_model(model, X, Y, epc=256, bs=8):
    model.fit(x=X, y=Y, epochs=epc, batch_size=bs, verbose=0, shuffle=True)
    return model


fitness_cache = {}


import ctypes
import platform


def lib_name(name):
    if(platform.system() == "Darwin"):
        ext = ".dylib"
    elif(platform.system() == "Linux"):
        ext = ".so"
    else:
        ext = ".dll"

    return name + ext


lut_types = {}
for radius in range(1, ca_max_radius+1):
    lut_types[radius] = ctypes.c_bool * ca_lut_len(radius)

gslcblas = ctypes.CDLL(lib_name('libgslcblas'), mode=ctypes.RTLD_GLOBAL)
libspatial = ctypes.CDLL(lib_name('libspatial'))
libspatial.lib_find_fitness.restype = ctypes.c_double


def tool_array_to_ctype(lut):
    return lut_types[ca_get_radius(lut)](*list(lut))


def evolve_fitness(lut, images, total_cell_count, max_gap=10):
    key = ''.join([str(x) for x in lut])
    try:
        return fitness_cache[key][1]
    except:
        fitness = libspatial.lib_find_fitness(
            tool_array_to_ctype(lut), ca_get_radius(lut))
        fitness_cache[key] = (lut, fitness)
        return fitness


def id_total_cell_count(images):
    non_gap = np.count_nonzero(images+1)
    x, _, z = images.shape
    return non_gap - x*z


def evolve_recalc_fitness(population, images, total_cell_count, max_gap=10):
    def fit(lut): return evolve_fitness(lut, images, total_cell_count, max_gap)
    fitness = np.array([fit(lut) for lut in population])
    return (fitness, fitness / np.sum(fitness))


def evolve_get_stats(fitness):
    return (np.min(fitness), np.mean(fitness), np.std(fitness), np.max(fitness))


def evolve_select_parent(population, pfitness):
    return population[np.random.choice(range(0, len(population)), p=pfitness)]


def evolve_build_new_individual(population, pfitness, pm):
    return evolve_mutate(evolve_cross(evolve_select_parent(population, pfitness), evolve_select_parent(population, pfitness)), pm)


def evolve_build_new_population(population, pfitness, pm, count):
    return [evolve_build_new_individual(population, pfitness, pm) for _ in range(0, count)]


def neural_unpack_cache():
    v = fitness_cache.values()
    return np.array([l[0] for l in v]), np.array([l[1] for l in v])


def evolve_global_elite(elite_size=4):
    X, Y = neural_unpack_cache()
    idxs = np.argsort(Y)
    return [X[idxs[-1*i]] for i in range(1, elite_size+1)]


def neural_build_model(model=None):
    if(not model):
        model = neural_get_model()
    X, Y = neural_unpack_cache()
    model = neural_train_model(model, X, Y)
    return model


def neural_recalc_fitness(model, population):
    X = np.array(population)
    fitness = model.predict(X).reshape(len(population))
    return (fitness, fitness / np.sum(fitness))


def evolve_algoritm(images=None, iterations=50000, pm=0.015, max_gap=10, population_size=256, elite_size=32):
    old_population = [ca_init_random(np.random.randint(
        ca_min_radius, ca_max_radius+1)) for _ in range(0, population_size)]

    if(images == None):
        total_cell_count = -1
    else:
        total_cell_count = id_total_cell_count(images)

    best = old_population[0]
    model = None
    model_age = 0
    for i in range(0, iterations):
        if(((not model) and (len(fitness_cache) > 2*population_size)) or (model_age > 200)):
            print("Building new neural net", end=" ")
            model_age = 0
            model = neural_build_model()
            print("Done!")
            pass

        if(model):
            fitness, pfitness = neural_recalc_fitness(model, old_population)
            model_age += 1
        else:
            fitness, pfitness = evolve_recalc_fitness(
                old_population, images, total_cell_count, max_gap)

        best = old_population[np.argmax(fitness)]

        new_population = evolve_build_new_population(
            old_population, pfitness, pm, population_size - elite_size)

        idxs = np.argsort(fitness)
        elite = [old_population[idxs[-1*i]]
                 for i in range(1, elite_size+1)]

        elite_fitness, _ = evolve_recalc_fitness(
            elite, images, total_cell_count, max_gap)

        print(i+1, evolve_get_stats(fitness),
              evolve_get_stats(elite_fitness), len(fitness_cache))

        if(np.max(elite_fitness) == 1.0):
            break

        old_population = new_population + elite

    return best


def test_make_spatial(vector, ps):
    l = vector.shape[0]
    r = np.random.random(l)
    for i in range(0, l):
        if (r[i] < ps):
            vector[i] = -1
    return vector


def test_get_image(lut, initial_conf, time_steps, max_gap=10, ps=0.0):
    result = np.array([initial_conf for _ in range(0, time_steps)])
    cur = initial_conf
    for t in range(1, time_steps):
        for _ in range(0, np.random.randint(1, max_gap+2)):
            cur = ca_apply(lut, cur)
        result[t] = test_make_spatial(cur, ps)
    return result


def test_prepare_images(lut, image_count, time_steps, cell_count, max_gap=10, ps=0.0):
    return np.array([test_get_image(lut, tool_random_binary_vector(cell_count), time_steps, max_gap, ps) for _ in range(0, image_count)], dtype=np.int8)


global_pows = np.array([ca_get_pows(r)
                        for r in range(ca_min_radius, ca_max_radius+1)])

import sys

target_rule = -1
try:
    target_rule = int(sys.argv[1])
except:
    print("Usage: python3 ./spatial.py [ECA rule number]")
    print("   ECA rule number = 0, 1, ..., 255")
    exit(-1)

print("Initializing C library...")
libspatial.lib_init(tool_array_to_ctype(
    ca_decode_rule(target_rule, 1)), 1, 64, 0)

print("Starting identfication algorithm")
best = evolve_algoritm()

print("BEST individual: ", best)
