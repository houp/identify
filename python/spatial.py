import numpy as np

ca_min_radius = 1
ca_max_radius = 5


def ca_get_radius(lut):
    return int((np.log2(lut.shape[0])-1)/2)


def ca_decode_rule(rule_num, radius):
    return np.array([int(c) for c in bin(rule_num)[2:].zfill(2**(2*radius+1))]).astype(np.int8)


def ca_init_random(radius):
    return np.random.random(2**(2*radius+1)).round().astype(np.int8)


def ca_downscale_lut(lut, check=True):
    if(check):
        min_radius = ca_min_radius
    else:
        min_radius = 1

    radiusBig = ca_get_radius(lut)

    if(radiusBig <= min_radius):
        return lut

    lenBig = lut.shape[0]
    lenSmall = lenBig / 4
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
    lenBigHalf = lenBig / 2
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
    pows = ca_get_pows(radius)
    input_len = input.shape[0]
    idxs = (pows * np.array([ca_conf_nei(input, i, radius)
                             for i in range(0, input_len)])).sum(1)
    return np.choose(idxs, lut)


def conf_diff(a, b):
    idxs = np.where(np.logical_and(a >= 0, b >= 0))[0]
    return np.abs(np.choose(idxs, a) - np.choose(idxs, b)).sum()


def conf_is_complete(conf):
    return conf.min() >= 0


def id_error_step(lut, input, target_output):
    output = ca_apply(lut, input)
    return (conf_diff(target_output, output), output)


def id_error_step_gap(lut, input, target_output, max_gap=10):
    error = np.zeros(max_gap)
    current_input = input
    min_error = input.shape[0]
    outputs = []

    for i in range(0, max_gap):
        e, current_input = id_error_step(lut, current_input, target_output)
        error[i] = e
        outputs.append(current_input)

    min_err = error.min()
    if(conf_is_complete(target_output)):
        return (min_err, outputs[error.argmin()])

    return (min_err, outputs[np.random.choice(np.where(error == min_err)[0])])
