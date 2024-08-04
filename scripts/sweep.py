import json
import argparse
import os

from scripts.train_cifar import train, trainConfig

def run_sweep(config_path, output_dir):

    with open(config_path, 'r') as f:
        config = json.load(f)

    configs = config['configs']

    for i, conf in enumerate(configs):
        print(f"\n>>>>>>>> STARTING SWEEP {i+1}/{len(configs)}: {conf.get('name')} <<<<<<<<\n")
        
        # User config overrides defaults
        run_args = conf.copy()
        run_args['output_dir'] = output_dir
        # os.path.join(output_dir, conf.get('name'))
        
        # Create TrainConfig
        cfg = trainConfig(run_args)
        
        # Execute
        try:
            train(cfg)
        except Exception as e:
            print(f"FAILED sweep {conf.get('name')}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='scripts/sweep.json')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    def is_debug_mode():
        import os
        import sys
        return (
            sys.gettrace() is not None or
            'pydevd' in sys.modules or
            'VSCODE_PID' in os.environ
        )
    if is_debug_mode():
        args.config_file= 'scripts/sweep_unified.json'
        args.output_dir = 'results_unified'

    run_sweep(args.config_file, args.output_dir)
