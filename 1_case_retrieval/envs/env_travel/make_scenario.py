import os,json,shutil,random
from pathlib import Path
from typing import Optional

scenario_travel = {
    "Sites":["Heritages", "Museum", "ArtGallery"],
    "Amusement":["Zoo", "Nightclub", "Parks"],
    "Meals":["LocalCuisine", "Cafe", "Bars"]
}




def generate_travel_preferences():


    issue_list = list(scenario_travel.keys())
    weights_values = [3, 2, 1]


    weights_a = random.sample(weights_values, len(issue_list))
    weights_b = random.sample(weights_values, len(issue_list))

    global_preference_a = {}
    global_preference_b = {}

    for i, issue in enumerate(issue_list):
        # random
        options = scenario_travel[issue]
        pref_a = random.sample(options, len(options))
        # must be different
        while True:
            pref_b = random.sample(options, len(options))
            if pref_b != pref_a:
                break

        global_preference_a[issue] = {
            "weight": weights_a[i],
            "options": pref_a
        }
        global_preference_b[issue] = {
            "weight": weights_b[i],
            "options": pref_b
        }

    return global_preference_a, global_preference_b


def make(
        scenario_file:str,
        ufun_file:str,
        output_path:str,
        pareto_file:Optional[str] = None,
        is_test:bool = False,
    ):

    folder = Path(os.path.dirname(scenario_file))
    
    with open(folder / 'map.json') as jf:
        alter_map = json.load(jf)
    
    with open(scenario_file, "r", encoding="utf-8") as f:
        scenario_text = f.read()
    name_a = random.sample(alter_map['ĻNAME_AĻ']['alternative'],k=1)[0]
    name_b = random.sample(alter_map['ĻNAME_BĻ']['alternative'],k=1)[0]
    cityname = random.sample(alter_map['ĻTRAVEL_DSTĻ']['alternative'],k=1)[0]

    scenario_text = scenario_text.replace('ĻNAME_AĻ',name_a).replace('ĻNAME_BĻ',name_b).replace('ĻTRAVEL_DSTĻ',cityname)

    with open(ufun_file, "r", encoding="utf-8") as f:
        ufun_text = f.read()

    pref_a, pref_b = generate_travel_preferences()
    with open(Path(output_path) / 'pref_a.json', 'w') as jf:
        jf.write(json.dumps(pref_a, indent=4))
    with open(Path(output_path) / 'pref_b.json', 'w') as jf:
        jf.write(json.dumps(pref_b, indent=4))

    sorted_issues_a = [issue for issue, _ in sorted(pref_a.items(), key=lambda x: x[1]['weight'], reverse=True)]
    sorted_issues_b = [issue for issue, _ in sorted(pref_b.items(), key=lambda x: x[1]['weight'], reverse=True)]


    # sorted_issues_a = ["\""+x+"\"" for x in sorted_issues_a]
    # sorted_issues_b = ["\""+x+"\"" for x in sorted_issues_b]


    scenario_text = scenario_text.replace('Ļa_DMN_AĻ',sorted_issues_a[0]).replace('Ļa_DMN_BĻ',sorted_issues_a[1]).replace('Ļa_DMN_CĻ',sorted_issues_a[2])
    scenario_text = scenario_text.replace('Ļb_DMN_AĻ',sorted_issues_b[0]).replace('Ļb_DMN_BĻ',sorted_issues_b[1]).replace('Ļb_DMN_CĻ',sorted_issues_b[2])

    scenario_text = scenario_text.replace('Ļa_DMN_A_prefĻ', ' > '.join(pref_a[sorted_issues_a[0]]['options']))
    scenario_text = scenario_text.replace('Ļa_DMN_B_prefĻ', ' > '.join(pref_a[sorted_issues_a[1]]['options']))
    scenario_text = scenario_text.replace('Ļa_DMN_C_prefĻ', ' > '.join(pref_a[sorted_issues_a[2]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_A_prefĻ', ' > '.join(pref_b[sorted_issues_b[0]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_B_prefĻ', ' > '.join(pref_b[sorted_issues_b[1]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_C_prefĻ', ' > '.join(pref_b[sorted_issues_b[2]]['options']))
    # preference statement
    scenario_text = scenario_text.replace('ĻDMN_SEQ_aĻ',f"You value Issue {sorted_issues_a[0]} the most, followed by Issue {sorted_issues_a[1]}, and finally Issue {sorted_issues_a[2]}.")
    scenario_text = scenario_text.replace('ĻDMN_SEQ_bĻ',f"You value Issue {sorted_issues_b[0]} the most, followed by Issue {sorted_issues_b[1]}, and finally Issue {sorted_issues_b[2]}.")
    # offer
    scenario_text = scenario_text.replace('Ļa_DMN_A_pref_listĻ', ' | '.join(pref_a[sorted_issues_a[0]]['options']))
    scenario_text = scenario_text.replace('Ļa_DMN_B_pref_listĻ', ' | '.join(pref_a[sorted_issues_a[1]]['options']))
    scenario_text = scenario_text.replace('Ļa_DMN_C_pref_listĻ', ' | '.join(pref_a[sorted_issues_a[2]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_A_pref_listĻ', ' | '.join(pref_b[sorted_issues_b[0]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_B_pref_listĻ', ' | '.join(pref_b[sorted_issues_b[1]]['options']))
    scenario_text = scenario_text.replace('Ļb_DMN_C_pref_listĻ', ' | '.join(pref_b[sorted_issues_b[2]]['options']))

    # default offer
    scenario_text = scenario_text.replace('ĻDEFAULT_a_AĻ', pref_a[sorted_issues_a[0]]['options'][-1])
    scenario_text = scenario_text.replace('ĻDEFAULT_a_BĻ', pref_a[sorted_issues_a[1]]['options'][-1])
    scenario_text = scenario_text.replace('ĻDEFAULT_a_CĻ', pref_a[sorted_issues_a[2]]['options'][-1])
    scenario_text = scenario_text.replace('ĻDEFAULT_b_AĻ', pref_b[sorted_issues_b[0]]['options'][-1])
    scenario_text = scenario_text.replace('ĻDEFAULT_b_BĻ', pref_b[sorted_issues_b[1]]['options'][-1])
    scenario_text = scenario_text.replace('ĻDEFAULT_b_CĻ', pref_b[sorted_issues_b[2]]['options'][-1])



    outputdir = Path(output_path)

    scenario_new = outputdir / os.path.basename(scenario_file)
    ufun_new = outputdir / os.path.basename(ufun_file)



    with open(scenario_new, 'w') as jf:
        jf.write(scenario_text)

    with open(ufun_new, 'w') as jf:
        jf.write(ufun_text)

    new_pareto_file = shutil.copy(folder/'pareto.py', outputdir)
    print(new_pareto_file)


    path = Path(new_pareto_file)
    (outputdir / "__init__.py").touch()

    import importlib
    import sys
    import pathlib

    scenario_dir = pathlib.Path(new_pareto_file).parent  


    sys.path.insert(0, str(scenario_dir))

    spec = importlib.util.spec_from_file_location("pareto_module", new_pareto_file)
    pareto_module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(pareto_module)


    pareto_module.main()

    pareto_new = outputdir / 'pareto.json'
    assert os.path.exists(pareto_new)
    


    return scenario_new, ufun_new, pareto_new

