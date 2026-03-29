from test_paper_reproduction import run_linking_pass_hypernodes
import utils_reproduction as reprutil
from utils.utils import Logger


log = Logger("zz")

temp_dir = "/home/asagirova/arigraph/cookbook_per_section_short_triplets_general_prompt_with_code/adaptive-pruning/repo"
code_files = reprutil.collect_code_files(temp_dir)
code_index = reprutil.build_code_index(temp_dir, code_files, log)
mem_graph = run_linking_pass_hypernodes(
    None, code_index, code_files, log, max_paper_triplets=None
)