def get_modifier(method: str, model_type: str = None):

    if method == "origin":
        from .origin import Origin
        return None, Origin
    
    if method == 'greedy':
        from .greedy_generation import Greedy
        return None, Greedy

    elif method == 'spotlight-train':
        from .spotlight_train import Spotlight
        return None, Spotlight
    
    elif method == 'spotlight-evaluation':
        from .spotlight_evaluation import Spotlight
        return None, Spotlight

    elif method == 'spotlight-generation':
        from .spotlight_generation import Spotlight
        return None, Spotlight
    
    elif method == 'linearhashing-train':
        from .linearhashing_train import LinearHashing
        return None, LinearHashing
    
    elif method == 'linearhashing-evaluation':
        from .linearhashing_evaluation import LinearHashing
        return None, LinearHashing
    
    elif method == 'linearhashing-generation':
        from .linearhashing_generation import LinearHashing
        return None, LinearHashing
    
    elif method == 'upperbound-evaluation':
        from .upperbound_evaluation import UpperBound
        return None, UpperBound
    
    elif method == 'upperbound-generation':
        from .upperbound_generation import UpperBound
        return None, UpperBound
    
    else:
        raise NotImplementedError(method, model_type)