def calc_reach_prob(node):
    """
    Calculate the arrival probability from the root node to the current node (node).
    The 'last_offer_accept_prob' at each level represents the probability of "accepting the previous offer".
    Thus, the probability of reaching the current node is the product of all the (1 - p_i) values from the upper levels.
    """

    prob = 1.0
    current = node
    
    while current.parent is not None:
        p_accept = getattr(current, "last_offer_accept_prob", None)
        if p_accept is None:
            raise ValueError(f"Node {current} has no attribute 'last_offer_accept_prob'")
        prob *= (1.0 - p_accept)
        current = current.parent
        
    return prob