import datetime

import optuna
from optuna.distributions import FloatDistribution

from my_optuna_study.core import History


# create study
str_time = datetime.datetime.now().strftime("%M%S")
study = optuna.create_study(
    storage="sqlite:///" + str_time + ".db",
    sampler=None,
    pruner=None,
    study_name="sample",
    direction=None,
    load_if_exists=False,
    directions=None,
)

# calc something
h = History()
h.construct(10, 5, 1)

# add trial
for i, row in h.df.iterrows():
    kwargs = dict(
        state=optuna.trial.TrialState.COMPLETE,
        params={k: v for k, v in zip(h.prm_names, row[h.prm_names])},
        distributions={k: FloatDistribution(0, 1) for k in h.prm_names},
        user_attrs=None,
        system_attrs=None,
        intermediate_values=None,
    )
    if len(h.obj_names) == 1:
        kwargs.update(dict(value=row[h.obj_names].values[0]))
    else:
        kwargs.update(dict(values=row[h.obj_names].values))
    trial = optuna.create_trial(**kwargs)
    study.add_trial(trial)

# visualize
fig = optuna.visualization.plot_optimization_history(
    study, target=None, target_name="Objective Value", error_bar=False
)
