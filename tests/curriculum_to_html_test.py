from adam.curriculum.phase1_curriculum import _make_each_object_by_itself_curriculum
from adam.curriculum_to_html import CurriculumToHtmlDumper


def test_simple_curriculum_html(tmp_path):
    instances = [_make_each_object_by_itself_curriculum()]
    html_dumper = CurriculumToHtmlDumper()
    html_dumper.dump_to_html(instances, output_directory=tmp_path, title="Test Objects")
