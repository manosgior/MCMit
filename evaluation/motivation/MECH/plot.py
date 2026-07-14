import matplotlib.pyplot as plt

# VQE benchmark counts
on_chip     = 532931
cross_chip  = 8035
meas_num    = 41358

# β values (x-axis) and α values (different lines)
beta_values  = [0.1, 0.5, 1, 2, 4, 6, 8, 10]
alpha_values = [2, 7, 10]

plt.figure(figsize=(8,5))
for alpha in alpha_values:
    eff_cnot = [
        on_chip + alpha * cross_chip + beta * meas_num
        for beta in beta_values
    ]
    plt.plot(beta_values, eff_cnot, marker='o', label=f"α = {alpha}")

plt.xscale('log')       # log scale for β
plt.xlabel("β (weight on measurement count)")
plt.ylabel("Effective CNOT count")
plt.title("Effective CNOTs vs β for various α (VQE)")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("plot.png")
