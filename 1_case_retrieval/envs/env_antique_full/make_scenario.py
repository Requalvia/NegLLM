import os,json,shutil,random
from pathlib import Path
from typing import Optional
def make(
        scenario_file:str,
        ufun_file:str,
        output_path:str,
        pareto_file:Optional[str] = None,
        is_test:bool = False,
    ):
    '''
    for antique environment, make a scenario 
    '''
    folder = Path(os.path.dirname(scenario_file))
    
    with open(folder / 'map.json') as jf:
        alter_map = json.load(jf)
    
    with open(scenario_file, "r", encoding="utf-8") as f:
        scenario_text = f.read()

    with open(ufun_file, "r", encoding="utf-8") as f:
        ufun_text = f.read()

    print(pareto_file)
    with open(pareto_file, "r", encoding="utf-8") as f:
        pareto_text = f.read()
    
    antique1 = random.choice(alter_map['ĻANTIQUE_1Ļ']['alternative'])
    antique2 = random.choice([x for x in alter_map['ĻANTIQUE_2Ļ']['alternative'] if x != antique1])
    deltax = random.choice(alter_map['ĻDELTA_XĻ']['alternative'])
    # shift prices

    price_keys = ["ĻPRICE_INITĻ", "ĻANT_1_INĻ", "ĻANT_2_INĻ", "ĻANT_1_OUTĻ", "ĻANT_2_OUTĻ"]

    scenario_text = scenario_text.replace('ĻANTIQUE_1Ļ', antique1).replace('ĻANTIQUE_2Ļ', antique2)
    ufun_text = ufun_text.replace('ĻANTIQUE_1Ļ', antique1).replace('ĻANTIQUE_2Ļ', antique2)

    pareto_text = pareto_text.replace('ĻANTIQUE_1Ļ', antique1).replace('ĻANTIQUE_2Ļ', antique2)

    for pk in price_keys:
        scenario_text = scenario_text.replace(pk, str(alter_map[pk]+deltax))
        ufun_text = ufun_text.replace(pk, str(alter_map[pk]+deltax))

        pareto_text = pareto_text.replace(pk, str(alter_map[pk]+deltax))


    pareto_json = json.loads(pareto_text)
    for x in pareto_json:
        x['a']+=deltax
        x['b']+=deltax

    outputdir = Path(output_path)

    scenario_new = outputdir / os.path.basename(scenario_file)
    ufun_new = outputdir / os.path.basename(ufun_file)


    pareto_new = outputdir / os.path.basename(pareto_file)

    with open(scenario_new, 'w') as jf:
        jf.write(scenario_text)

    with open(ufun_new, 'w') as jf:
        jf.write(ufun_text)


    with open(pareto_new, 'w') as jf:
        jf.write(json.dumps(pareto_json,indent=4)) 
    

    return scenario_new, ufun_new, pareto_new


