import json
import matplotlib.pyplot as plt
import numpy as np

with open('data.json', 'r') as f:
    loaded_data = json.load(f)
# Visualize the best policy
#agents.visualize_best()
steps_evaluation = loaded_data["steps_evaluation"]
all_taux_accord = loaded_data["all_taux_accord"]
minsize = min(len(sublist) for sublist in steps_evaluation)
steps_evaluation = steps_evaluation[0][0:minsize]

minsize = min(len(sublist) for sublist in all_taux_accord)
all_taux_accord = [sublist[0:minsize]for sublist in all_taux_accord]

steps_evaluation = np.array(steps_evaluation)
all_taux_accord = np.array(all_taux_accord)

mean_steps = np.mean(all_taux_accord, axis=0)
std_steps = np.std(all_taux_accord, axis=0)
print(mean_steps.shape)
print(std_steps.shape)

plt.plot(steps_evaluation, mean_steps)
plt.fill_between(steps_evaluation, mean_steps - std_steps, mean_steps + std_steps, 
                     color="b", alpha=0.2, label="Écart-type")
plt.xlabel("steps")
plt.ylabel("taux d'accord")
plt.title("Taux d'accord entre l'actor et le critic lors de chaque évaluation")
plt.show()