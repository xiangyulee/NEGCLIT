from util.prune import prune_channel,prune
model_weight_lighting={'channel':prune_channel,'default':prune}
from experiment.offline_train import auto_split,fixed_split,self_grow
offline_train_method={'autosplit':auto_split,'fixedsplit':fixed_split,'selfgrow':self_grow}
from experiment.deploy_method import autosplit_deploy,selfgrow_deploy,fixedsplit_deploy
deploy_method={'autosplit':autosplit_deploy,'selfgrow':selfgrow_deploy,'fixedsplit':fixedsplit_deploy}
from experiment.online_train import online_autosplit,online_fixedsplit,online_selfgrow
online_train_method={'autosplit':online_autosplit,'fixedsplit':online_fixedsplit,'selfgrow':online_selfgrow}