from adam.curriculum.phase1_curriculum import EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM
from adam.curriculum_to_html import CurriculumToHtmlDumper


def test_simple_curriculum_html(tmp_path):
    instances = [EACH_OBJECT_BY_ITSELF_SUB_CURRICULUM]
    html_dumper = CurriculumToHtmlDumper()
    html_dumper.dump_to_html(instances, output_destination=tmp_path, title="Test Objects")
