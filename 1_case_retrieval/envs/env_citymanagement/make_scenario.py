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

    folder = Path(os.path.dirname(scenario_file))
    with open(scenario_file, "r", encoding="utf-8") as f:
        scenario_text = f.read()

    with open(ufun_file, "r", encoding="utf-8") as f:
        ufun_text = f.read()

    alpha = random.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    beta = random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])



    ufun_text = ufun_text.replace('ĻALPHAĻ', str(alpha))
    ufun_text = ufun_text.replace('ĻBETAĻ', str(beta))


    outputdir = Path(output_path)

    scenario_new = outputdir / os.path.basename(scenario_file)
    ufun_new = outputdir / os.path.basename(ufun_file)

    with open(scenario_new, 'w') as jf:
        jf.write(scenario_text)

    with open(ufun_new, 'w') as jf:
        jf.write(ufun_text)

    new_pareto_file = shutil.copy(folder/'pareto.py', outputdir)
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