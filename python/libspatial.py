import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import ctypes
import platform
import sys
import os

ca_min_radius = 2
ca_max_radius = 2
fitness_cache = {}
max_best_age = 250

def lib_name(name):
    if(platform.system() == "Darwin"):
        ext = ".dylib"
    elif(platform.system() == "Linux"):
        ext = ".so"
    else:
        ext = ".dll"

    return name + ext


gslcblas = ctypes.CDLL(lib_name('libgslcblas'), mode=ctypes.RTLD_GLOBAL)
pathname = os.path.dirname(sys.argv[0])
libspatial = ctypes.CDLL(lib_name(os.path.abspath(pathname)+'/libspatial'))
libspatial.lib_find_fitness.restype = ctypes.c_double


def ca_get_radius(lut):
    return int((np.log2(lut.shape[0])-1)/2)


def ca_lut_len(radius):
    return 2**(2*radius+1)


def ca_decode_rule(rule_num, radius):
    return np.array([int(c) for c in bin(rule_num)[2:].zfill(ca_lut_len(radius))]).astype(np.int8)


def ca_decode_rule_bool(rule_num, radius):
    return [bool(c) for c in bin(rule_num)[2:].zfill(ca_lut_len(radius))]


def tool_random_binary_vector(lenght):
    result = np.zeros(lenght)
    for i in range(0,lenght):
        r = np.random.rand()
        if(np.random.rand() < r):
            result[i]=1
    return result.astype(np.int8)


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


def neural_get_model(afun='tanh', layer_count=2):
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


lut_types = {}
for radius in range(1, ca_max_radius+1):
    lut_types[radius] = ctypes.c_bool * ca_lut_len(radius)


def tool_array_to_ctype(lut):
    return lut_types[ca_get_radius(lut)](*list(lut))


def evolve_cached_fitness(lut):
    key = ''.join([str(x) for x in lut])
    try:
        return fitness_cache[key][1]
    except:
        return None


def evolve_fitness(lut):
    fitness = evolve_cached_fitness(lut)
    if(not fitness):
        fitness = libspatial.lib_find_fitness(
            tool_array_to_ctype(lut), ca_get_radius(lut))
        fitness_cache[''.join([str(x) for x in lut])] = (lut, fitness)
    return fitness


def evolve_recalc_fitness(population):
    fitness = np.array([evolve_fitness(lut) for lut in population])
    return (fitness, fitness / np.sum(fitness))


def evolve_get_stats(fitness):
    return (np.min(fitness), np.mean(fitness), np.std(fitness), np.max(fitness))


def evolve_select_parent(population, pfitness):
    return population[np.random.choice(range(0, len(population)), p=pfitness)]


def evolve_build_new_individual(population, pfitness, pm):
    return evolve_mutate(
        evolve_cross(evolve_select_parent(population, pfitness), 
                     evolve_select_parent(population, pfitness)), pm)


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

    # TODO: check if this makes sense
    for i in range(0, len(population)):
        cached = evolve_cached_fitness(population[i])
        if(cached != None):
            fitness[i] = cached
   
    return (fitness, fitness / np.sum(fitness))


def evolve_algoritm(iterations=50000, base_pm=0.015, max_gap=10, population_size=128, elite_size=8):
    old_population = [ca_init_random(np.random.randint(
        ca_min_radius, ca_max_radius+1)) for _ in range(0, population_size)]

    best = old_population[0]
    best_age = 0
    best_fitness = 0.0

    model = None
    model_age = 0

    for i in range(0, iterations):
        # dynamic mutation probability dependent on the best age
        pm = np.exp(-0.025 * (max_best_age - best_age))

        if (pm > 1.0):
            pm = 1.0
        if (pm < base_pm):
            pm = base_pm
   
        # restart
        if(best_age > max_best_age):
            best_age = 0
            best_fitness = 0.0
            model = None
            model_age = 0
            print("Reset of population!")
            old_population = [ca_init_random(np.random.randint(ca_min_radius, ca_max_radius+1)) for _ in range(0, population_size)]

        # build neural model
        if(((not model) and (len(fitness_cache) > 6*population_size))):
            print("Building new neural net", end=" ")
            model_age = 0
            model = neural_build_model()
            print("Done!")
            pass

        if(model_age > 250):
            model = None

        # counting fitness values
        if(model):
            fitness, pfitness = neural_recalc_fitness(model, old_population)
            model_age += 1
        else:
            fitness, pfitness = evolve_recalc_fitness(old_population)

        best = old_population[np.argmax(fitness)]

        tmpmax = np.max(fitness)
        if(tmpmax > best_fitness):
            best_fitness = tmpmax
            best_age = 0
        else:
            best_age += 1

        new_population = evolve_build_new_population(
            old_population, pfitness, pm, population_size - elite_size)

        idxs = np.argsort(fitness)
        elite = [old_population[idxs[-1*i]]
                 for i in range(1, elite_size+1)]

        elite_fitness, _ = evolve_recalc_fitness(elite)

        print(i+1, pm, best_age, evolve_get_stats(fitness),
              evolve_get_stats(elite_fitness), len(fitness_cache))

        if(np.max(elite_fitness) == 1.0):
            best = elite[np.argmax(elite_fitness)]
            break

        old_population = new_population + elite

    return best


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
