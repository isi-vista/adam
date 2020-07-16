from adam.curriculum.phase1_curriculum import _make_each_object_by_itself_curriculum
from adam.curriculum_to_html import CurriculumToHtmlDumper
from adam.language_specific.chinese.chinese_language_generator import (
    GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)


def test_simple_curriculum_html(tmp_path):
    instances = [
        _make_each_object_by_itself_curriculum(
            None, None, GAILA_PHASE_1_LANGUAGE_GENERATOR
        )
    ]
    html_dumper = CurriculumToHtmlDumper()
    html_dumper.dump_to_html(instances, output_directory=tmp_path, title="Test Objects")


def test_simple_curriculum_html_chinese(tmp_path):
    instances = [
        _make_each_object_by_itself_curriculum(
            None, None, GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR
        )
    ]
    html_dumper = CurriculumToHtmlDumper()
    html_dumper.dump_to_html(instances, output_directory=tmp_path, title="Test Objects")
