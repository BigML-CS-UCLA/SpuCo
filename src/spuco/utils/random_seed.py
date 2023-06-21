import random 

class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance

def set_seed(new_seed: int):
    """
    Set the seed for the SpuCo module.

    :param new_seed: The new seed value to set.
    :type new_seed: int
    """
    seed_class = Singleton()
    seed_class.seed = new_seed 

def get_seed():
    """
    Get the seed value of the SpuCo module.

    :return: The seed value.
    :rtype: int
    """
    seed_class = Singleton()
    if not hasattr(seed_class, "seed"):
        return random.randint(0, 10000000)
    else:
        return seed_class.seed
    
def seed_randomness(random_module=None, torch_module=None, numpy_module=None):
    """
    Seed the randomness of the specified modules.

    :param random_module: The random module. Default is None.
    :type random_module: Optional[ModuleType]
    :param torch_module: The torch module. Default is None.
    :type torch_module: Optional[ModuleType]
    :param numpy_module: The numpy module. Default is None.
    :type numpy_module: Optional[ModuleType]
    """
    seed = get_seed()
    if torch_module is not None:
        torch_module.backends.cudnn.deterministic = True 
        torch_module.backends.cudnn.benchmark = False 
        torch_module.manual_seed(seed)
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    if random_module is not None:
        random_module.seed(seed)