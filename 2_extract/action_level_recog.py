from copy import deepcopy
import json
from typing import Any, Dict, List, Tuple

def traverse_for_pairs(node: Dict[str, Any],
                       main_viewer: str,
                       ctx: List[Tuple[str, str]],
                       system_prompts: Dict[str, str],
                       tu_prompt: str,
                       offer_gen_prompts: Dict[str, str],
                       offer_decision_prompts: list,
                       samples: List[Dict[str, Any]],
                       last_tu_sample=None):

    role = node.get("role", "?")
    if role not in ("a", "b"):
        return

    role_label = "You" if role == main_viewer else "Opponent"

    combined = build_combined_text(node)
    new_ctx = ctx + [(role_label, combined)] if combined else ctx
    if len(new_ctx) > 6:
        new_ctx = new_ctx[-6:]

    children = node.get("children", [])
    if not children:
        return


    if role != main_viewer and len(children) >= 2:


        max_fv = max(float(c.get(f"future_value_{main_viewer}", 0.0)) for c in children)
        strong_threshold = 0.75 * max_fv

        good_children = [
            c for c in children
            if float(c.get(f"future_value_{main_viewer}", 0.0)) >= strong_threshold
        ]

        good_children = sorted(
            good_children,
            key=lambda x: float(x.get(f"future_value_{main_viewer}", 0.0)),
            reverse=True
        )

        total_fv = sum(float(c.get(f"future_value_{main_viewer}", 0.0)) for c in good_children)

        sys_for_role = system_prompts.get(main_viewer, "")
        dialogue_history = build_dialogue_history(node, main_viewer)
        current_depth = node.get("depth", 0)
        total_depth = node.get("max_depth", 16)

        system_text = (
            f"{sys_for_role}\n"
            f"The dialogue history is\n{dialogue_history}\n\n"
            f"This is the {current_depth} round of total {total_depth} negotiation rounds."
        )


        for ch in good_children:

            future_value = float(ch.get(f"future_value_{main_viewer}", 0.0))
            normalized_value = future_value / total_fv if total_fv > 0 else 0
            

            if ch.get("thought") and ch.get("utterance"):
                output_tu = {
                    "thought": ch["thought"],
                    "utterance": ch["utterance"]
                }

                tu_sample = {
                    "instruction": tu_prompt,
                    "input": "",
                    "output": json.dumps(output_tu, ensure_ascii=False),
                    "system": system_text,
                    "history": [],
                    "reward": ch[f"future_value_{main_viewer}"],
                    'normalized_value' : normalized_value
                }
                samples.append(tu_sample)
                last_tu_sample = tu_sample

            if ch.get("offer") is not None:
                sys_for_offer = offer_gen_prompts.get(main_viewer, "")
                offer_system_text = (
                    f"{sys_for_offer}\n"
                    f"This is the {current_depth} round of total {total_depth} negotiation rounds."
                )

                if last_tu_sample:
                    offer_input = last_tu_sample["output"]
                    offer_history = [[last_tu_sample["instruction"], last_tu_sample["output"]]]
                else:
                    offer_input = ""
                    offer_history = []

                output_offer = {"offer": ch["offer"]}

                offer_sample = {
                    "instruction": sys_for_offer,
                    "input": offer_input,
                    "output": json.dumps(output_offer, ensure_ascii=False),
                    "system": offer_system_text,
                    "history": offer_history,
                    "reward": ch[f"future_value_{main_viewer}"],
                    'normalized_value' : normalized_value
                }
                samples.append(offer_sample)


    max_fv_node = max(float(c.get(f"future_value_{main_viewer}", 0.0)) for c in children)
    traverse_threshold = 0.75 * max_fv_node

    filtered_children = [
        c for c in children
        if float(c.get(f"future_value_{main_viewer}", 0.0)) >= traverse_threshold
    ]

    filtered_children = sorted(
        filtered_children,
        key=lambda c: float(c.get(f"future_value_{main_viewer}", 0.0)),
        reverse=True
    )

    for c in filtered_children:
        traverse_for_pairs(c, main_viewer, new_ctx,
                           system_prompts, tu_prompt,
                           offer_gen_prompts,
                            offer_decision_prompts, samples, last_tu_sample)



def build_combined_text(node: Dict[str, Any]) -> str:
    t = node.get("thought", "")
    u = node.get("utterance", "")
    o = node.get("offer", None)
    offer_part = f" Offer: {o}" if o is not None else ""
    return f"Thought: {t} Utterance: {u}{offer_part}".strip()


def build_dialogue_history(node: Dict[str, Any], view: str) -> str:
    if view not in ['a', 'b']:
        raise ValueError("view must be 'a' or 'b'")

    path = []
    current = node
    while current:
        path.append(current)
        current = current.get("parent", None)
    path.reverse()

    dialogue_lines = []
    for n in path:
        if not n.get("utterance"):
            continue
        offer_str = f" Offer: {n['offer']}" if n.get("offer") is not None else ""
        prefix = "You" if n.get("role") == view else "Opponent"
        dialogue_lines.append(f"{prefix}: Utterance: {n['utterance']}{offer_str}")
    return "\n".join(dialogue_lines)






def convert_tree_to_llamafactory(system_prompts: Dict[str, str],
                                 tu_prompt: str,
                                 offer_gen_prompts: Dict[str, str],
                                 offer_decision_prompts: list,
                                 data: Dict[str, Any],
                                 main_viewer: str,
                                 out_path: str):
    samples: List[Dict[str, Any]] = []

    traverse_for_pairs(data, main_viewer, [], system_prompts,
                       tu_prompt, offer_gen_prompts, offer_decision_prompts, samples, None)



    return samples
