import os,json,shutil
from pathlib import Path
import importlib.util
import importlib


def make_scenario(
        env_dir:str,
        scenario_file:str,
        ufun_file:str,
        pareto_file:str,
        output_path:str,
    ):
    '''
    If the 'make_scenario.py' exists under the source scene folder, it indicates customization is needed.
    Otherwise, simply copy it.
    Returns the path to the new scenario / ufun / pareto files.
    '''


    assert os.path.exists(env_dir)
    custom_script = Path(env_dir) / 'make_scenario.py'
    snr_path = Path(output_path) / 'scenario'
    os.mkdir(snr_path)
    
    if os.path.exists( custom_script ) is False:
        scenario_file = shutil.copy(scenario_file, snr_path)
        ufun_file = shutil.copy(ufun_file, snr_path)
        pareto_file = shutil.copy(pareto_file, snr_path)
        return scenario_file, ufun_file, pareto_file


    spec = importlib.util.spec_from_file_location("custom_make_scenario", custom_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "make"):
        raise AttributeError(f"'make_scenario.py' must define a function named 'make'.")


    scenario_file, ufun_file, pareto_file = module.make(
        scenario_file=scenario_file,
        ufun_file=ufun_file,
        pareto_file=pareto_file,
        output_path=snr_path,
        is_test=False
    )
    return scenario_file, ufun_file, pareto_file