from util.prune import prune_channel,prune
model_weight_lighting={'channel':prune_channel,'default':prune}
from experiment.train_method import auto_split,fixed_split,self_grow
train_method={'autosplit':auto_split,'fixedsplit':fixed_split,'selfgrow':self_grow}