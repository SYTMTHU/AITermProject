import pickle
import numpy

f_input = open('1316input', 'rb')
f_target = open('1316target', 'rb')


reload_input = pickle.load(f_input)
reload_target = pickle.load(f_target)
assert(len(reload_input) == len(reload_target))
length = len(reload_input)
tencile = int(length/10)
candidate = [i for i in range(length)]
validNtest = numpy.random.choice(candidate, size = 4*tencile, replace=False)
train = list(set(candidate) - set(validNtest))

valid = numpy.random.choice(validNtest, size = 2*tencile, replace=False)

test = list(set(validNtest) - set(valid))


f_input_test = open('input_test', 'wb')
f_input_valid = open('input_valid', 'wb')
f_input_train = open('input_train', 'wb')

f_target_test = open('target_test', 'wb')
f_target_valid = open('target_valid', 'wb')
f_target_train = open('target_train', 'wb')

def fromtotaltodivided(original, indexlist, f):
    result = [original[i] for i in indexlist]
    pickle.dump(result, f)
    f.close()

fromtotaltodivided(reload_input, train, f_input_train)
fromtotaltodivided(reload_target, train, f_target_train)

fromtotaltodivided(reload_input, valid, f_input_valid)
fromtotaltodivided(reload_target, valid, f_target_valid)

fromtotaltodivided(reload_input, test, f_input_test)
fromtotaltodivided(reload_target, test, f_target_test)