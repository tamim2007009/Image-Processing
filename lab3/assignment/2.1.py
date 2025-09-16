# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 22:24:01 2025

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def erlang_pdf(x, k, mu):
   
    return (x**(k-1) * np.exp(-x/mu)) / (mu**k * math.factorial(k-1))


if __name__ == "__main__":
 
    k = int(input("Enter shape parameter k (positive integer): "))
    mu = float(input("Enter scale parameter μ (positive real number): "))

    # Generate x values (range of intensities 0–255 like histogram bins)
    x = np.linspace(0, 255, 256)

   
    pdf = erlang_pdf(x, k, mu)

    # Normalize PDF to make it sum to 1 (like a histogram)
    pdf = pdf / np.sum(pdf)

    # Plot target histogram
    plt.figure(figsize=(8,5))
    plt.bar(x, pdf, width=1.0, color="skyblue", edgecolor="black")
    plt.title(f"Erlang Target Histogram (k={k}, μ={mu})")
    plt.xlabel("Intensity Value")
    plt.ylabel("Probability")
    
    plt.show()
