from util.method_match import offline_train_method,deploy_method,online_train_method

def offline_run(args,NE=None):
    offline_train_method[args.train_method](args,NE=NE)
    
def deploy(args):
    return deploy_method[args.train_method](args)

def online_run(args,NE=None):
    online_train_method[args.train_method](args,NE=NE)







