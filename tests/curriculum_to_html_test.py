from adam.curriculum.phase1_curriculum import EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM
from adam.curriculum_to_html import CurriculumToHtml


def test_simple_curriculum_html():
    instances = [EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM]
    CurriculumToHtml.generate(instances, "./", overwrite=True, title="Test Objects")
