from tests.interaction_test import run_interaction_test
from tests.overall_test import run_overall_tests, run_wa_we_test, run_pqr_test
from src.util.graph_format import all_to_weka


if __name__ == "__main__":
    # all_to_weka()
    run_overall_tests()
    # run_interaction_test()
    # run_wa_we_test()
    # run_pqr_test()
