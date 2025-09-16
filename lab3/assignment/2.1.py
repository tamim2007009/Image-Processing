import numpy as np
import matplotlib.pyplot as plt
import math

# Erlang PDF using scale mu
def erlang_pdf(x, k, mu):
    x = np.maximum(x, 0.01)  # avoid zero
    return (x**(k-1) * np.exp(-x/mu)) / (mu**k * math.factorial(k-1))

# Histogram function
def erlang_histogram(k, mu, L=256):
    hist = np.zeros(L, dtype=np.float32)

    # Scale x to capture the peak (approx 2*mean)
    max_x = max(20, 2 * k * mu)

    for i in range(L):
        x_val = i * max_x / 255.0
        hist[i] = erlang_pdf(x_val, k, mu)

    # Normalize histogram to sum = 1
    hist = hist / hist.sum()

    return hist

# Main execution
if __name__ == "__main__":
    k = int(input("Enter shape parameter k (positive integer): "))
    mu = float(input("Enter scale parameter μ (positive real number): "))

    hist = erlang_histogram(k, mu)
    x_bins = np.arange(len(hist))

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.bar(x_bins, hist, width=1.0, color="skyblue", edgecolor="black")
    plt.title(f"Erlang Histogram (k={k}, μ={mu})")
    plt.xlabel("Intensity Value")
    plt.ylabel("Probability")
    plt.show()
