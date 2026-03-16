from typing import List, Dict, Optional
from numpy import mean
from typing_extensions import LiteralString
import uuid
import math
import json


class ShortTermNode:
    def __init__(self,
                 role:str,
                 step: Optional[int] = -1,
                 max_depth :Optional[int] = 16,
                 
                 thought: Optional[str] = "",
                 utterance: Optional[str] = "",
                 offer:Optional[Dict] = {},
                 is_terminal:Optional[bool] = False,
                 
                 style_prob: Optional[Dict[str, Dict[str, float]]] = None,
                 parent: Optional['ShortTermNode'] = None,

                 type: Optional[str] = 'natural',
                 last_offer_accept_prob: Optional[float] = 0.00,
                 prob: Optional[float] = None,
                 value: Optional[float] = None):


        assert role in ['a','b','unexplored']

        self.id = str(uuid.uuid4()) 
        self.role=role
        self.step = step
        self.depth = step
        self.max_depth = max_depth
        self.utterance = utterance
        self.thought = thought
        self.offer = offer
        self.style_prob = style_prob if style_prob is not None else {}

        self.parent = parent
        self.children: List[ShortTermNode] = []

        self.prob = prob
        self.value = value

        self.is_terminal = is_terminal
        self.last_offer_accept_prob=last_offer_accept_prob
        

        # mcts stats
        # self.visit_count = 0
        self.total_value = []
        """
        A class representing a node in a short-term decision tree for negotiation or dialogue processes.

        Attributes:
        -----------
        id : str
            A unique identifier for the node.
        role : str
            The role of the agent at this node, can be 'a', 'b', or 'unexplored'.
        step : int
            The step in the negotiation or dialogue process at this node.
        depth : int
            The depth of the node in the tree (same as step by default).
        max_depth : int
            The maximum allowable depth of the negotiation or dialogue tree.
        thought : str
            The internal thought or reasoning behind this node's state.
        utterance : str
            The message or utterance made by the agent at this node.
        offer : dict
            A dictionary representing the offer made by the agent (key-value pairs of items and their values).
        is_terminal : bool
            A flag indicating whether this node is a terminal (end) node.
        style_prob : dict
            A dictionary containing probabilities for different conversational styles and strategies.
        parent : ShortTermNode
            A reference to the parent node of this node in the tree.
        children : list of ShortTermNode
            A list of child nodes of this node.
        prob : float
            The probability associated with this node's state.
        value : float
            The utility or value of this node in the context of the decision-making process.
        last_offer_accept_prob : float
            The probability of accepting the last offer made.
        total_value : list
            A list tracking the cumulative value of this node, typically used for Monte Carlo Tree Search (MCTS) statistics.
        """

    def get_visit_count(self,):
        return len(self.total_value)
    
    def get_avg_rewards(self,):
        if len(self.total_value) == 0:
            return 0.0
        return mean(self.total_value)



    def add_child(self, child_node: 'ShortTermNode'):
        child_node.parent = self
        self.children.append(child_node)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0



    def to_dict(self) -> Dict:
        if self.role == 'unexplored':
            return {"role": self.role}
        return {
            "id": self.id,
            "role": self.role,
            "step": self.step,
            'depth':self.depth,
            'max_depth':self.max_depth,
            "thought": self.thought,
            "utterance": self.utterance,
            "offer":self.offer,
            "style_prob": self.style_prob,
            "total_value": self.total_value,
            "ALP":self.last_offer_accept_prob,
            "children": [child.to_dict() for child in self.children]
        }
    
    def short_offer(self)->str:
        if self.role == 'unexplored':
            return self.role
        vs =",".join([str(v) for k,v in self.offer.items()]) 
        return "<"+vs+">"

    def __repr__(self) -> str:
        if self.role == 'unexplored':
            return self.role
        end1 = " END" if self.is_terminal else ''
        return f"step {self.step} ({self.role}) ALP:{round(self.last_offer_accept_prob,3)} {self.short_offer()} visit_count:{self.get_visit_count()},r={[round(float(x),2) for x in self.total_value]} {end1}"

    def find_paths_to_root(self) -> list:
        """
        from some node to root, get full path
        """
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    

    def build_dialogue_history(self, view: str) -> str:
        """
        Build the full dialogue history from the root to the current node,
        formatted as a readable text string for LLM input.

        :param view: current perspective ("a" or "b")
                    - view == 'a': a's own messages are "You:", opponent's are "Opponent:"
                    - view == 'b': b's own messages are "You:", opponent's are "Opponent:"
        :return: str
                e.g.,
                You: Utterance: Hello. Offer: {Coin: No, Painting: 3000}
                Opponent: Utterance: I'd prefer 4000. Offer: {Coin: No, Painting: 4000}
        """
        if view not in ['a', 'b']:
            raise ValueError("view must be 'a' or 'b'")

        # backtrack to root
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        path.reverse()

        # make dialogue lines
        dialogue_lines = []
        for node in path:
            if not node.utterance:
                continue

            # Offer
            offer_str = ""
            if node.offer is not None:
                offer_str = f" Offer: {node.offer}"

            # self or opponent
            if node.role == view:
                prefix = "You"
                content = f"Utterance: {node.utterance}{offer_str}"
            else:
                prefix = "Opponent"
                content = f"Utterance: {node.utterance}{offer_str}"

            dialogue_lines.append(f"{prefix}: {content}")

        dialogue_lines = dialogue_lines[-6:]
        # one dialogue text
        dialogue_text = "\n".join(dialogue_lines)
        return dialogue_text






class NegotiationTree:
    """
    a tree structure to manage ShortTermNode nodes

    """
    def __init__(self,  root_node: ShortTermNode, main_viewer: str):

        self.root = root_node
        self.nodes = [root_node] 
        self.node_map = {root_node.step: [root_node]}  
        self.main_viewer = main_viewer

    def add_node(self, node: ShortTermNode, parent: ShortTermNode):

        node.parent = parent
        parent.children.append(node)
        self.nodes.append(node)
        self.node_map.setdefault(node.step, []).append(node)

    def remove_node(self, node: ShortTermNode):
        """
        Delete a leaf node.
        If the parent node becomes an empty node (no children and not a terminal node) after the deletion, recursively delete the parent node.
        - Only supports deletion starting from leaf nodes;
        - Automatically updates parent.children / self.nodes / self.node_map;
        - Returns the number of nodes deleted.
        """

        removed_count = 0

        def _remove_recursive(current_node):
            nonlocal removed_count

            
            if current_node.children:
                return False

            parent = current_node.parent

            
            if parent is not None:
                parent.children = [c for c in parent.children if c is not current_node]

            
            if current_node in self.nodes:
                self.nodes.remove(current_node)

            if current_node.step in self.node_map:
                self.node_map[current_node.step] = [
                    n for n in self.node_map[current_node.step] if n is not current_node
                ]
                if not self.node_map[current_node.step]:
                    del self.node_map[current_node.step]

            
            current_node.parent = None
            current_node.children = []
            removed_count += 1

            # check parent node for recursive deletion
            if (
                parent is not None
                and len(parent.children) == 0
                and not parent.is_terminal
            ):

                _remove_recursive(parent)

        _remove_recursive(node)
        return removed_count
    
    
    def get_nodes_by_step(self, step: int) -> list:
        """
        get all nodes at a specific step
        """
        return self.node_map.get(step, [])

    def get_latest_nodes(self) -> list:
        """
        get all nodes at the maximum step
        """
        if not self.node_map:
            return []
        max_step = max(self.node_map.keys())
        return self.node_map[max_step]

    def find_paths_to_root(self, node: ShortTermNode) -> list:
        """
        backtrack from some node to root, get full path
        """
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_leaf_nodes(self) -> list[ShortTermNode]:
        """
        get all leaf nodes in the tree
        """
        return [node for node in self.nodes if not node.children]
    

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"NegotiationTree(role={self.role}, total_nodes={len(self.nodes)})"

    def save_to_json(self, file_path: str):
        """
        save the entire tree to a JSON file
        """
        tree_dict = self.root.to_dict()  
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, ensure_ascii=False, indent=2)
    
    def print_structure(self, max_depth: Optional[int] = None):
        """
        print the tree structure
        """
        def _print_subtree(node: ShortTermNode, prefix: str = "", is_last: bool = True, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return

            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{node}  "

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _print_subtree(child, child_prefix, i == len(node.children) - 1, depth + 1)

        
        _print_subtree(self.root)
