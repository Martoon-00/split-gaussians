import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
# Generation
#

def generate(args, size):
    def genLinear(mean, std):
        return np.random.normal(mean, std)
    red = [np.vectorize(np.random.normal)(args.red_mean, args.red_std) for _ in range(0, size)]
    blue = [np.vectorize(np.random.normal)(args.blue_mean, args.blue_std) for _ in range(0, size)]
    return red, blue


def me(red, blue, iterations=5, apriori=None, test_callback=None):
    both_colors = np.concatenate((red, blue))
    np.random.shuffle(both_colors)

    one_blue = blue[0]
    one_red = red[0]

    if apriori:
        apriori.weight *= np.sqrt(both_colors.size)

    print "ME is here"
    print "Learning points:", len(both_colors)
    print "Iteration:", iterations
    if apriori:
        print "Using MAP with weight", apriori.weight
    else:
        print "No MAP"
    print

    #
    # Initial estimates
    #

    red_mean_guess = np.min(both_colors)
    blue_mean_guess = np.max(both_colors)

    dispersion =  np.sqrt(reduce(np.add, both_colors ** 2) / both_colors.size -
                         (reduce(np.add, both_colors) / both_colors.size) ** 2) \

    if apriori:
        apriori_std = dispersion / np.sqrt(apriori.weight)

    red_std_guess = dispersion / 5
    blue_std_guess = dispersion / 5

    #
    # Preliminary plot
    #

    red_line, = plt.plot(np.transpose(red[1:])[0], np.transpose(red[1:])[1], 'ro', alpha=0.2)
    blue_line, = plt.plot(np.transpose(blue[1:])[0], np.transpose(blue[1:])[1], 'bo', alpha=0.2)
    redl_line, = plt.plot(one_red[0], one_red[1], 'r^')
    bluel_line, = plt.plot(one_blue[0], one_blue[1], 'b^')

    plt.legend([red_line, blue_line, redl_line, bluel_line], \
                ['Red data', 'Blue data', 'Red labeled', 'Blue labeled'])
    # plt.show(block=False)

    #
    # Main loop
    #

    for i in range(0, iterations):
        print "Iteration #" + str(i)

        # count data points likelihood
        likelihood_of_red = map(np.product, stats.norm(red_mean_guess, red_std_guess).pdf(both_colors))
        likelihood_of_blue = map(np.product, stats.norm(blue_mean_guess, blue_std_guess).pdf(both_colors))

        # count data points weights
        likelihood_total = np.add(likelihood_of_red, likelihood_of_blue)

        red_weight = np.divide(likelihood_of_red, likelihood_total)
        blue_weight = np.divide(likelihood_of_blue, likelihood_total)

        if test_callback:
            suppose_red = [x for (x, p) in zip(both_colors, red_weight) if p > 0.5]
            test_callback(suppose_red)

        # evalute next approximation of guess
        def estimate_mean(data, weight):
            return np.dot(weight, data) / np.sum(weight)

        def estimate_mean_with_apriori(data, weight, apriori_mean, std):
            from_data = np.dot(weight, np.matrix(data)).tolist()[0]
            numerator = np.add(std ** 2 * apriori_mean, apriori_std ** 2 * from_data)
            denominator = std ** 2 + np.sum(weight) * apriori_std ** 2
            return np.divide(numerator, denominator)

        def estimate_std(data, weight, mean):
            variance = np.dot(weight, np.power(data - mean, 2)) / np.sum(weight)
            return np.sqrt(variance)

        # new estimates for standard deviation
        red_std_guess = estimate_std(both_colors, red_weight, red_mean_guess)
        blue_std_guess = estimate_std(both_colors, blue_weight, blue_mean_guess)

        # new estimates for mean
        if apriori:
            red_mean_guess = estimate_mean_with_apriori(both_colors, red_weight, one_red, red_std_guess)
            blue_mean_guess = estimate_mean_with_apriori(both_colors, blue_weight, one_blue, blue_std_guess)
        else:
            red_mean_guess = estimate_mean(both_colors, red_weight)
            blue_mean_guess = estimate_mean(both_colors, blue_weight)

        print "Guesses:"
        print "Red:", "mean", red_mean_guess, "std", red_std_guess
        print "Blue:", "mean", blue_mean_guess, "std", blue_std_guess
        print

    #
    # Gathering fun
    #

    plt.close()

    red_line, = plt.plot(np.transpose(red[1:])[0], np.transpose(red[1:])[1], 'ro', alpha=0.2)
    blue_line, = plt.plot(np.transpose(blue[1:])[0], np.transpose(blue[1:])[1], 'bo', alpha=0.2)
    redl_line, = plt.plot(one_red[0], one_red[1], 'r^')
    bluel_line, = plt.plot(one_blue[0], one_blue[1], 'b^')

    plt.legend([red_line, blue_line, redl_line, bluel_line], \
                ['Red data', 'Blue data', 'Red labeled', 'Blue labeled'])
    reds_line, = plt.plot(red_mean_guess[0], red_mean_guess[1], 'rs')
    blues_line, = plt.plot(blue_mean_guess[0], blue_mean_guess[1], 'bs')
    plt.legend([red_line, blue_line, redl_line, bluel_line, reds_line, blues_line], \
                ['Red data', 'Blue data', 'Red labeled', 'Blue labeled', 'Red guess', 'Blue guess'])
    plt.show(block=False)
    plt.pause(4)
    plt.close()

## Returns function, which for given set of points returns function,
## which accepts another set and returns number of mismatches.
def testExaminer(red):
    def f(supposed_red):
        def diff_size(l1, l2):
            missed = 0
            for x in l2:
                not_found = max(0, 1 - len([1 for y in l1 if (x == y).all()]))
                missed += not_found
            return missed
        errors = diff_size(red, supposed_red) + diff_size(supposed_red, red)
        error_rate = float(errors) / (len(red) + len(blue))
        print "Error rate: ", error_rate * 100, "%"
        if (error_rate > 0.2):
            inv_errors = diff_size(blue, supposed_red) + diff_size(supposed_red, blue)
            print "Error rate inversed: ", float(inv_errors) / (len(red) + len(blue)) * 100, "%"
    return f


np.random.seed(7075) # for reproducible random results

gen = lambda: None
gen.red_mean = [10, 0]
gen.red_std = [6, 6]

gen.blue_mean = [0, 10]
gen.blue_std = [6, 6]

print "Generating:"
print "Red:", "mean", gen.red_mean, "std", gen.red_std
print "Blue:", "mean", gen.blue_mean, "std", gen.blue_std
print

red, blue = generate(gen, size = 100)

testAndPrint = testExaminer(red)

# apriori_params = None
apriori_params = lambda: None
apriori_params.weight = 1

me(red, blue, apriori=apriori_params, test_callback=testAndPrint, iterations=10)

# for w in range(0, 100):
#     apriori_params.weight = w
#     me(red, blue, apriori = apriori_params)

