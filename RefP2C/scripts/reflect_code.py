import os
import sys
import logging
import argparse
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.configs.config import LLM_MODELS
from src.utils.helper import read_file, save_code, sanitize_code
from src.reflection.verifier import CodeVerifier
from src.reflection.revision_planner import RevisionPlanner
from src.reflection.editor import CodeEditor
from src.reflection.controller import RefinementController

logging.basicConfig(format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Code Reflection and Refinement Pipeline")
    
    parser.add_argument(
        '--workspace_dir', 
        type=str, 
        default="results/default", 
        help="Directory to save all outputs for this run."
    )
    
    parser.add_argument('--paper_id', type=str, default='', help="The unique ID of the paper.")
    parser.add_argument('--model_eval', type=str, default="gpt-4o-mini")
    parser.add_argument('--model_plan', type=str, default="gpt-4o-mini")
    parser.add_argument('--model_revise', type=str, default="gpt-4o-mini")
    parser.add_argument('--max_attempts', type=int, default=1, help="Max refinement cycles.")
    
    return parser.parse_args()

class CodeReflectionPipeline:
    def __init__(self, args):
        self.args = args
        self._setup_and_validate_paths()
        
        verifier = CodeVerifier(model=LLM_MODELS[args.model_eval], max_workers=5)
        planner = RevisionPlanner(self.workspace_dir, model=LLM_MODELS[args.model_plan])
        editor = CodeEditor(self.workspace_dir, model=LLM_MODELS[args.model_revise])
        
        self.controller = RefinementController(self.workspace_dir, verifier, planner, editor)

    def _setup_and_validate_paths(self):
        self.workspace_dir = os.path.join(PROJECT_ROOT, self.args.workspace_dir)
        logging.info(f"🚀 Starting Code Reflection. Using workspace: {self.workspace_dir}")
        
        self.initial_code_dir = os.path.join(self.workspace_dir, "repo", "initial_repo")
        self.signals_path = os.path.join(self.workspace_dir, "signal_design", "supervisory_signals_final.json")
        self.paper_path = os.path.join(PROJECT_ROOT, 'paper', self.args.paper_id, 'paper.md')
        
        if not os.path.isdir(self.initial_code_dir):
            raise FileNotFoundError(f"Initial code directory not found at the expected location: {self.initial_code_dir}")
        if not os.path.exists(self.signals_path):
            raise FileNotFoundError(f"Supervisory signals file not found at the expected location: {self.signals_path}")
        if not os.path.exists(self.paper_path):
            raise FileNotFoundError(f"Paper markdown file not found for ID {self.args.paper_id} at: {self.paper_path}")

    def run(self):
        logging.info("--- Stage 1: Loading all input data ---")
        try:
            expected_file_order = ['config.yaml', 'main.py', 'experiments.py']
            
            initial_project = {}
            
            for filename in expected_file_order:
                file_path = os.path.join(self.initial_code_dir, filename)
                if os.path.exists(file_path):
                    initial_project[filename] = read_file(file_path)
                else:
                    logging.error(f"Expected project file not found: {file_path}. Halting.")
                    sys.exit(1)
            with open(self.signals_path, 'r', encoding='utf-8') as f:
                supervisory_signals = json.load(f)
            paper_content = read_file(self.paper_path)
        except Exception as e:
            logging.error(f"Failed to load input data: {e}", exc_info=True)
            sys.exit(1)

        final_project = self.controller.run_refinement_cycle(
            initial_project=initial_project,
            criteria_data=supervisory_signals,
            paper_text=paper_content,
            max_major_attempts=self.args.max_attempts
        )

        final_code_dir = os.path.join(self.workspace_dir, "repo/final_repo")
        os.makedirs(final_code_dir, exist_ok=True)
        for filename, content in final_project.items():
            save_path = os.path.join(final_code_dir, filename)
            save_code(sanitize_code(content), save_path)
            
        logging.info("="*50)
        logging.info("🎉🎉🎉 Code Reflection Pipeline COMPLETED SUCCESSFULLY! 🎉🎉🎉")
        logging.info(f"Find the final refined code and detailed logs in: {final_code_dir}")
        logging.info("="*50)

def main():
    try:
        args = parse_args()
        pipeline = CodeReflectionPipeline(args)
        pipeline.run()
    except Exception as e:
        logging.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)
        sys.exit(1)
        
if __name__ == '__main__':
    main()