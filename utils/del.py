import math

def kendalls_tau(rank1, rank2):
    if len(rank1) != len(rank2):
        raise ValueError("Both rankings must have the same number of elements.")
    
    n = len(rank1)
    num_concordant = 0
    num_discordant = 0
    items = list(rank1.keys())
    print(items)
    
    for i in range(n-1):
        for j in range(i+1, n):
            item1 = items[i]
            item2 = items[j]
            print(item1, item2)
            a = rank1[item1] - rank1[item2]
            b = rank2[item1] - rank2[item2]
            if a * b > 0:
                num_concordant += 1
            elif a * b < 0:
                num_discordant += 1
    
    tau = (num_concordant - num_discordant) / (0.5 * n * (n - 1))
    return tau

# Example rankings
rank1 = {'a': 1, 'b': 2, 'c': 3}
rank2 = {'a': 3, 'b': 1, 'c': 2}

# Calculate Kendall's Tau
kendall_tau = kendalls_tau(rank1, rank2)
print(f"Kendall's Tau: {kendall_tau}")
