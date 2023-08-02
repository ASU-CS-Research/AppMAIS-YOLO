import matplotlib.pyplot as plt

# make a function that takes in x and y as arrays and plots them
def plot(x, y, x_label, y_label, title):
    plt.plot(x, y, 'bo', ms=8, label='betabinom pmf')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def get_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            if "n_true" in line:
                x.append(float(line.split(":")[1]))
            if "log likelihood beta" in line:
                y.append(float(line.split(":")[1]))
                continue
        return x, y

if __name__ == '__main__':
    filename = "fixed_betabinomial_out_11s.txt"

    x, y = get_data(filename)
    plot(x, y, "n_true", "log_likelihood", "log likelihood vs n_true")

