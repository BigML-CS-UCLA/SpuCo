import random 

class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance

def set_seed(new_seed: int):
    """
    Set seed for SpuCo module.
    """
    seed_class = Singleton()
    seed_class.seed = new_seed 

def get_seed():
    seed_class = Singleton()
    if not hasattr(seed_class, "seed"):
        return random.randint(0, 10000000)
    else:
        return seed_class.seed
    
def seed_randomness(random_module=None, torch_module=None, numpy_module=None):
    seed = 0 #get_seed()
    if torch_module is not None:
        torch_module.backends.cudnn.deterministic = True 
        torch_module.backends.cudnn.benchmark = False 
        torch_module.manual_seed(seed)
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    if random_module is not None:
        random_module.seed(seed)