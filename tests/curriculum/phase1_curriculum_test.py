from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM


def test_instantiate_curriculum():
    for sub_curriculum in GAILA_PHASE_1_CURRICULUM:
        for _ in sub_curriculum.instances():
            # we don't need to do anything
            # the curriculum may be dynamically generated
            # so we just want to test we can instantiate it
            pass
