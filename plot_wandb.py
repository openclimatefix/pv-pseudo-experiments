import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.ticker as ticker

api = wandb.Api()

run = api.run("openclimatefix/PvMetNet/ekar03uc")

mae_names = [f"MAE_forecast_horizon_{i}/val_epoch" for i in range(360)]
epoch_values = []
for name in mae_names:
    epoch_values.append(run.summary[name])

print(np.mean(epoch_values))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch_values)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*5))
ax.xaxis.set_major_formatter(ticks_x)
fig.suptitle("MAE per forecast horizon")
ax.set_xlabel("Forecast horizon (minutes)")
ax.set_ylabel("MAE")
plt.savefig("epoch_mae_per_forecast_horizon.png")
plt.show()

mae_names = [f"MAE_forecast_horizon_{i}/val_step" for i in range(360)]
mae_values = defaultdict(list)
for name in mae_names:
    for i, row in run.history().iterrows():
        if row[name] != np.nan:
            if row[name] is not None:
                mae_values[name].append(row[name])
# Get average of each mae_name and plot them together on plot
mae_values = {k: np.mean(v) for k, v in mae_values.items()}
mae_values = [v for k, v in sorted(mae_values.items(), key=lambda item: int(item[0].split("_")[-1]))]
print(np.mean(mae_values))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mae_values)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*5))
ax.xaxis.set_major_formatter(ticks_x)
fig.suptitle("MAE per forecast horizon")
ax.set_xlabel("Forecast horizon (minutes)")
ax.set_ylabel("MAE")
plt.savefig("step_mae_per_forecast_horizon.png")
plt.show()
